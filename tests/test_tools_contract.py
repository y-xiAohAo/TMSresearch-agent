#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具契约测试：schema 完整性与返回结构。

- web_search / tms_optimize / s4l_run_script：真实环境冒烟（缺依赖自动跳过）
- sim4life_manual_qa：mock SSE，验证拼接与错误处理
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from research_agent.tools import (
    ALL_DESCRIPTORS,
    check_requirements,
    s4l_script,
    sim4life_manual_qa,
    tms_optimize,
    web_search,
)


class DescriptorContractTests(unittest.TestCase):
    def test_all_descriptors_have_required_metadata(self):
        self.assertGreaterEqual(len(ALL_DESCRIPTORS), 5)
        for desc in ALL_DESCRIPTORS:
            self.assertIn(desc.category, ("literature", "simulation", "compute", "knowledge", "verify"))
            self.assertIn(desc.cost_hint, ("free", "cheap", "expensive"))
            self.assertIsInstance(desc.requires, list)
            fmt = desc.spec.to_openai_format()
            self.assertEqual(fmt["type"], "function")
            self.assertIn("name", fmt["function"])
            self.assertIn("parameters", fmt["function"])

    def test_check_requirements_returns_dict(self):
        missing = check_requirements(strict=False)
        self.assertIsInstance(missing, dict)


class WebSearchContractTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("os").getenv("TAVILY_API_KEY") or _has_env_key(),
        "需要 TAVILY_API_KEY",
    )
    def test_real_search_returns_results(self):
        result = web_search._web_search("streamfunction TMS coil design", max_results=3)
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)
        item = result["results"][0]
        self.assertIn("title", item)
        self.assertIn("url", item)
        self.assertIn("content", item)


def _has_env_key() -> bool:
    from research_agent.config import SETTINGS

    return bool(SETTINGS.tavily_api_key)


class ManualQaContractTests(unittest.TestCase):
    def test_healthcheck_raises_when_service_down(self):
        with patch.object(sim4life_manual_qa.requests, "get", side_effect=ConnectionError("refused")):
            with self.assertRaises(RuntimeError) as ctx:
                sim4life_manual_qa._sim4life_manual_qa("test")
            self.assertIn("RAG 服务不可达", str(ctx.exception))

    def test_answer_tokens_joined(self):
        sse_lines = [
            "event: meta",
            'data: {"effective_model": "deepseek-v4-flash"}',
            "",
            "event: answer_token",
            'data: {"text": "第一步："}',
            "",
            "event: answer_token",
            'data: {"text": "点击 New。"}',
            "",
            "event: done",
            'data: {"ok": true}',
            "",
        ]

        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = iter(sse_lines)
        fake_resp.raise_for_status = MagicMock()
        fake_resp.__enter__ = lambda s: s
        fake_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch.object(sim4life_manual_qa, "_healthcheck", lambda: None),
            patch.object(sim4life_manual_qa.requests, "post", return_value=fake_resp),
        ):
            result = sim4life_manual_qa._sim4life_manual_qa("如何创建项目？")

        self.assertEqual(result["answer"], "第一步：点击 New。")


class S4lScriptSmokeTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("pathlib").Path(
            "D:/Sim4life/sim4lifeprogram/Sim4LifeLight_7.0.0.7995/Python/python.exe"
        ).is_file(),
        "需要本机 Sim4Life 安装",
    )
    def test_headless_create_and_save(self):
        import tempfile
        from pathlib import Path

        out = Path(tempfile.mkdtemp(prefix="s4l_smoke_")) / "t.smash"
        body = (
            "import s4l_v1.document as doc\n"
            "doc.New()\n"
            "import s4l_v1.model as model\n"
            "s = model.CreateSolidSphere(model.Vec3(0.0,0.0,0.0), 10.0)\n"
            "s.Name = 'smoke_sphere'\n"
            "doc.SaveAs(r'%s')\n"
            "print('ENTITIES', len(model.AllEntities()))\n" % str(out).replace("\\", "\\\\")
        )
        script = Path(tempfile.mkdtemp(prefix="s4l_script_")) / "smoke.py"
        script.write_text(s4l_script.build_script(body), encoding="utf-8")
        result = s4l_script._s4l_run_script(str(script), timeout_s=240)
        self.assertEqual(result["exit_code"], 0, msg=result["stderr"])
        self.assertTrue(out.is_file(), f"产物未生成: {result['stdout']} {result['stderr']}")


class TmsOptimizeContractTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("pathlib").Path(
            "D:/myvibeproject/StreamFunctionTMS/.venv/Scripts/python.exe"
        ).is_file(),
        "需要 StreamFunctionTMS venv",
    )
    def test_small_optimization_runs(self):
        result = tms_optimize._tms_optimize(
            {"Nsteps": 2, "pop_size": 10, "n_gen": 2},
            budget_s=180,
        )
        self.assertEqual(result["status"], "ok", msg=str(result["metrics"])[:500])
        self.assertIn("execution_time_sec", result["metrics"])
        self.assertGreater(result["metrics"].get("n_pareto_solutions", 0), 0)


if __name__ == "__main__":
    unittest.main()
