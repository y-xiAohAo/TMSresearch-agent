#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""literature 层测试：实体 / client(mock+真实) / pdf / extract / 工具契约。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from research_agent.literature.client import ArxivClient, normalize_arxiv_id
from research_agent.literature.extract import extract_sim_params
from research_agent.literature.pdf import _parse_page_range

_SAMPLE_XML = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2511.00744v1</id>
    <title>  Magnetic Materials for TMS  </title>
    <published>2025-11-02T00:03:49Z</published>
    <updated>2025-11-02T00:03:49Z</updated>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Li</name></author>
    <arxiv:primary_category term="physics.med-ph" />
    <category term="physics.med-ph" />
    <category term="eess.SP" />
    <link href="https://arxiv.org/pdf/2511.00744v1" rel="related" type="application/pdf" title="pdf"/>
    <summary>  Various coils for TMS are widely available. We propose a figure-8 coil of radius 0.05 m.  </summary>
  </entry>
</feed>"""


def _client_no_net() -> ArxivClient:
    import tempfile

    return ArxivClient(cache_dir=Path(tempfile.mkdtemp(prefix="arxiv_cache_")), min_interval=0)


class NormalizeIdTests(unittest.TestCase):
    def test_strip_version(self):
        self.assertEqual(normalize_arxiv_id("2511.00744v2"), ("2511.00744", "v2"))

    def test_prefix_and_default_version(self):
        self.assertEqual(normalize_arxiv_id("arxiv:2301.00001"), ("2301.00001", "v1"))

    def test_no_version(self):
        self.assertEqual(normalize_arxiv_id("2511.00744"), ("2511.00744", "v1"))


class ClientParsingTests(unittest.TestCase):
    def test_parse_entry_fields(self):
        client = _client_no_net()
        import xml.etree.ElementTree as ET

        root = ET.fromstring(_SAMPLE_XML)
        entry = root.findall("{http://www.w3.org/2005/Atom}entry")[0]
        p = client._parse_entry(entry)
        self.assertEqual(p.arxiv_id, "2511.00744")
        self.assertEqual(p.version, "v1")
        self.assertEqual(p.title, "Magnetic Materials for TMS")
        self.assertEqual(p.authors, ["Alice Zhang", "Bob Li"])
        self.assertIn("physics.med-ph", p.categories)
        self.assertIn("eess.SP", p.categories)
        self.assertTrue(p.pdf_url)
        self.assertIn("figure-8 coil", p.abstract)

    def test_search_uses_cache(self):
        client = _client_no_net()
        with patch.object(client, "_get", return_value=_SAMPLE_XML) as mock_get:
            first = client.search("tms", max_results=1)
            second = client.search("tms", max_results=1)
        self.assertEqual(mock_get.call_count, 1, "第二次应命中缓存")
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].arxiv_id, second[0].arxiv_id)

    def test_throttle_respects_interval(self):
        import tempfile, time

        client = ArxivClient(cache_dir=Path(tempfile.mkdtemp()), min_interval=0.2)
        client._throttle()
        start = time.monotonic()
        client._throttle()
        self.assertGreaterEqual(time.monotonic() - start, 0.18)

    def test_fetch_error_entry_returns_none(self):
        # arXiv 报错 entry 无作者 → 判 not_found
        client = _client_no_net()
        error_xml = _SAMPLE_XML.replace(
            '<author><name>Alice Zhang</name></author>', ""
        ).replace('<author><name>Bob Li</name></author>', "")
        with patch.object(client, "_get", return_value=error_xml):
            self.assertIsNone(client.fetch("0000.00000"))

    def test_error_titled_real_paper_not_mistaken_as_missing(self):
        # 标题以 "Error" 开头但有作者的真论文不得误判为不存在
        client = _client_no_net()
        error_titled = _SAMPLE_XML.replace(
            "Magnetic Materials for TMS", "Error bounds for coil design"
        )
        with patch.object(client, "_get", return_value=error_titled):
            paper = client.fetch("2511.00744")
        self.assertIsNotNone(paper)
        self.assertEqual(paper.title, "Error bounds for coil design")

    def test_non_xml_response_raises_runtime_error(self):
        client = _client_no_net()
        with patch.object(client, "_get", return_value="Service temporarily overloaded, retry later"):
            with self.assertRaises(RuntimeError) as ctx:
                client.search("tms")
            self.assertIn("无法解析", str(ctx.exception))

    def test_download_pdf_sanitizes_old_style_id(self):
        import tempfile

        client = _client_no_net()
        dest_dir = Path(tempfile.mkdtemp())
        pdf_bytes = b"%PDF-1.4 fake content"
        with patch("research_agent.literature.client.requests.get") as mock_get:
            resp = MagicMock()
            resp.content = pdf_bytes
            resp.raise_for_status = MagicMock()
            mock_get.return_value = resp
            path = client.download_pdf("hep-th/9901001", dest_dir=dest_dir)
        self.assertTrue(path.is_file())
        self.assertNotIn("/", path.name.replace(".pdf", ""))
        self.assertIn(dest_dir.resolve(), path.resolve().parents)

    def test_download_pdf_rejects_path_escape(self):
        import tempfile

        client = _client_no_net()
        dest_dir = Path(tempfile.mkdtemp())
        with self.assertRaises(RuntimeError):
            client.download_pdf("../../evil", dest_dir=dest_dir)


class PageRangeTests(unittest.TestCase):
    def test_range(self):
        self.assertEqual(_parse_page_range("1-3"), [1, 2, 3])

    def test_list_and_single(self):
        self.assertEqual(_parse_page_range("2,5"), [2, 5])
        self.assertEqual(_parse_page_range("4"), [4])

    def test_dedupe(self):
        self.assertEqual(_parse_page_range("1,1,2"), [1, 2])

    def test_reverse_range_raises(self):
        with self.assertRaises(ValueError):
            _parse_page_range("3-1")

    def test_fallback(self):
        self.assertEqual(_parse_page_range("abc"), [1])


class ExtractionTests(unittest.TestCase):
    _PAPER = (
        "We propose a figure-8 coil of radius 0.05 m positioned 2cm above head. "
        "The target field strength is 2.0 T at 0.02 m depth in motor cortex. "
        "Optimization used NSGA2 with pop_size 40 and n_gen 60 for focus."
    )

    _GOOD = {
        "coil_geometry": {"type": "figure8", "radius_m": 0.05, "position": "2cm above head"},
        "target_field": {"strength_T": 2.0, "focal_depth_m": 0.02, "region": "motor cortex"},
        "simulation": {"solver": None, "mesh_cells": None, "boundary": None},
        "algorithm": {"name": "NSGA2", "pop_size": 40, "n_gen": 60, "objectives": ["focus"]},
        "evidence_quotes": {
            "coil_geometry.type": "figure-8 coil",
            "coil_geometry.radius_m": "radius 0.05 m",
            "coil_geometry.position": "2cm above head",
        },
        "confidence": "high",
    }

    def test_valid_payload_becomes_entity(self):
        result = extract_sim_params(self._PAPER, lambda msgs: json.dumps(self._GOOD))
        from research_agent.literature.models import SimParamExtraction

        self.assertIsInstance(result, SimParamExtraction)
        self.assertEqual(result.coil_geometry.type, "figure8")
        self.assertEqual(result.coil_geometry.radius_m, 0.05)
        self.assertEqual(result.confidence, "high")

    def test_json_block_extracted_from_prose(self):
        # LLM 输出带前后散文时，raw_decode 应只取第一个完整 JSON 对象
        prose = "这是抽取结果：" + json.dumps(self._GOOD) + " 希望对你有帮助，{详见上文}"
        result = extract_sim_params(self._PAPER, lambda msgs: prose)
        self.assertEqual(result.coil_geometry.radius_m, 0.05)

    def test_llm_infrastructure_error_propagates_not_mislabeled(self):
        # P0-1 回归：llm_chat 抛异常必须向上传播，不得误报 extraction_failed，且不得 NameError
        def failing_llm(msgs):
            raise ConnectionError("api down")

        with self.assertRaises(ConnectionError):
            extract_sim_params("paper", failing_llm)

    def test_evidence_coverage_enforced(self):
        # P1-6：非 null 字段缺引句 → 校验失败
        payload = dict(self._GOOD)
        payload["evidence_quotes"] = {}  # 清空引句
        result = extract_sim_params(self._PAPER, lambda msgs: json.dumps(payload), max_retries=0)
        self.assertEqual(result["status"], "extraction_failed")
        self.assertIn("引句", result["raw"])

    def test_fabricated_quote_rejected(self):
        # P1-6：引句不是原文子串 → 判编造
        payload = dict(self._GOOD)
        payload["evidence_quotes"] = {
            "coil_geometry.type": "a completely invented sentence not in paper",
            "coil_geometry.radius_m": "radius 0.05 m",
            "coil_geometry.position": "2cm above head",
        }
        result = extract_sim_params(self._PAPER, lambda msgs: json.dumps(payload), max_retries=0)
        self.assertEqual(result["status"], "extraction_failed")
        self.assertIn("非原文子串", result["raw"])

    def test_null_fields_for_unmentioned(self):
        payload = json.dumps({
            "coil_geometry": {"type": None, "radius_m": None, "position": None},
            "target_field": {"strength_T": None, "focal_depth_m": None, "region": None},
            "simulation": {"solver": None, "mesh_cells": None, "boundary": None},
            "algorithm": {"name": None, "pop_size": None, "n_gen": None, "objectives": None},
            "evidence_quotes": {},
            "confidence": "low",
        })
        result = extract_sim_params("paper", lambda msgs: payload)
        self.assertIsNone(result.target_field.strength_T)
        self.assertIsNone(result.algorithm.name)

    def test_coil_geometry_null_object_accepted(self):
        # P1-7 回归：coil_geometry 整体为 null 不得 AttributeError
        payload = json.dumps({
            "coil_geometry": None,
            "target_field": {"strength_T": None, "focal_depth_m": None, "region": None},
            "simulation": {"solver": None, "mesh_cells": None, "boundary": None},
            "algorithm": {"name": None, "pop_size": None, "n_gen": None, "objectives": None},
            "evidence_quotes": {},
            "confidence": "low",
        })
        result = extract_sim_params("paper", lambda msgs: payload)
        from research_agent.literature.models import SimParamExtraction

        self.assertIsInstance(result, SimParamExtraction)

    def test_invalid_json_retries_then_fails(self):
        calls = []

        def bad_llm(msgs):
            calls.append(1)
            return "not json at all"

        result = extract_sim_params("paper", bad_llm, max_retries=1)
        self.assertEqual(result["status"], "extraction_failed")
        self.assertEqual(len(calls), 2, "应重试 1 次")

    def test_invalid_enum_rejected(self):
        bad = json.dumps({
            "coil_geometry": {"type": "hexagon", "radius_m": None, "position": None},
            "target_field": {}, "simulation": {}, "algorithm": {},
            "evidence_quotes": {}, "confidence": "high",
        })
        result = extract_sim_params("paper", lambda msgs: bad, max_retries=0)
        self.assertEqual(result["status"], "extraction_failed")


class ToolContractTests(unittest.TestCase):
    def test_descriptors(self):
        from research_agent.tools import arxiv_fetch, arxiv_read_pdf, arxiv_search, lit_extract_params

        for desc in (
            arxiv_search.DESCRIPTOR,
            arxiv_fetch.DESCRIPTOR,
            arxiv_read_pdf.DESCRIPTOR,
            lit_extract_params.DESCRIPTOR,
        ):
            self.assertEqual(desc.category, "literature")
            fmt = desc.spec.to_openai_format()
            self.assertEqual(fmt["type"], "function")


def _net_ok() -> bool:
    import socket

    try:
        socket.create_connection(("export.arxiv.org", 443), timeout=5).close()
        return True
    except OSError:
        return False


@unittest.skipUnless(_net_ok(), "需要 arXiv 网络可达")
class RealArxivTests(unittest.TestCase):
    def test_real_search(self):
        from research_agent.tools import arxiv_search

        result = arxiv_search._arxiv_search(
            "transcranial magnetic stimulation", field="ti", max_results=2
        )
        self.assertEqual(result["status"], "ok")
        self.assertGreater(len(result["papers"]), 0)
        self.assertIn("arxiv_id", result["papers"][0])

    def test_real_fetch(self):
        from research_agent.tools import arxiv_fetch

        result = arxiv_fetch._arxiv_fetch("2511.00744")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["paper"]["abstract"])


if __name__ == "__main__":
    unittest.main()
