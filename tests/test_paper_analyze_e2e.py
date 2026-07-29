#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_analyze 真实论文端到端（skipUnless 网络；断言结构性不变量，适配 LLM 非确定性）。"""

from __future__ import annotations

import unittest


def _net_ok() -> bool:
    import socket

    try:
        socket.create_connection(("export.arxiv.org", 443), timeout=5).close()
        return True
    except OSError:
        return False


@unittest.skipUnless(_net_ok(), "需要 arXiv 网络可达")
class RealPaperAnalyzeTests(unittest.TestCase):
    def test_real_paper_structural_invariants(self):
        from research_agent.tools import paper_analyze

        out = paper_analyze._paper_analyze("2511.00744", focus="simulation_params")

        # 结构性不变量（不断言必须 ok——flash 模型非确定性）
        self.assertIn(out["status"], ("ok", "incomplete", "extraction_failed"))
        pages = out["provenance"]["pages_read"]
        self.assertTrue(any(p > 2 for p in pages), f"应读到 Methods 段（>2 页），实际 {pages}")

        if out["status"] == "ok":
            result = out["result"]
            self.assertIsNotNone(result)
            # 引句必须为原文子串（闸 3 已强制，这里再独立验证一次）
            from research_agent.literature import paper_cache
            from research_agent.literature.client import normalize_arxiv_id

            norm_id, version = normalize_arxiv_id("2511.00744")
            cached = paper_cache.load(norm_id, version)
            self.assertIsNotNone(cached)
            full = " ".join(paper_cache.full_text({int(k): v for k, v in cached["page_texts"].items()}).split())
            for field_key, quote in (result.get("evidence_quotes") or {}).items():
                if quote:
                    self.assertIn(" ".join(str(quote).split()), full, f"引句不在原文：{field_key}")


if __name__ == "__main__":
    unittest.main()
