#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_model 工具层单测：mock s4l_script 子进程层，断言 overrides 组装与验证行为。

不启动真实 S4L 进程；编译走真实 sim4life compiler（纯文本生成）。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.literature.synthesis import get_compiler
from research_agent.literature.synthesis import s4l_compiler
from research_agent.tools import s4l_model


class _Harness:
    """包住一次 _s4l_model 调用：spy compiler + mock 写脚本/执行层。"""

    def __init__(self, test: unittest.TestCase, tmp_dir: str):
        self.captured: dict = {}
        self._smash = str(Path(tmp_dir) / "model_task.smash")
        real_compiler = get_compiler("sim4life")

        def _spy(task, overrides):
            self.captured["overrides"] = dict(overrides or {})
            out = real_compiler(task, overrides)
            self.captured["expected"] = out["expected"]
            self.captured["body"] = out["script_body"]
            return out

        def _fake_write(body, filename="s4l_model_task.py"):
            return {"script_path": str(Path(tmp_dir) / filename)}

        def _fake_run(script_path, timeout_s=600):
            Path(self._smash).touch()
            exp = self.captured["expected"]
            lines = [f"REPORT|ENTITY_COUNT|{exp['entity_count_min']}"]
            lines += [f"REPORT|ENTITY|{n}" for n in exp["entity_names"]]
            lines += [f"REPORT|MATERIAL|{n}|{m}"
                      for n, m in (exp.get("materials") or {}).items()]
            lines.append("REPORT|DONE")
            body = self.captured["body"]
            if "REPORT|PREFLIGHT|OK" in body:
                lines.append("REPORT|PREFLIGHT|OK")
            if "REPORT|SOLVE|HasResults|" in body:
                lines.append("REPORT|SOLVE|HasResults|True")
            return {"exit_code": 0, "stdout": "\n".join(lines),
                    "stderr": "", "duration_s": 0.1}

        self._patches = [
            patch.object(s4l_model, "get_compiler", lambda name: _spy),
            patch.object(s4l_compiler, "_default_smash_path", lambda: self._smash),
            patch.object(s4l_model.s4l_script, "_s4l_write_script", _fake_write),
            patch.object(s4l_model.s4l_script, "_s4l_run_script", _fake_run),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


class S4lModelToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="s4l_model_test_")
        self.addCleanup(self._tmp.cleanup)

    def _run(self, **kwargs):
        with _Harness(self, self._tmp.name) as h:
            result = s4l_model._s4l_model(radius=0.05, wire_diameter_mm=2.0, **kwargs)
        return result, h.captured

    def test_legacy_geometry_only_zero_change(self):
        result, captured = self._run()
        self.assertEqual(result["status"], "ok", msg=str(result["verification"]))
        # 旧调用方式不引入任何新 override
        self.assertEqual(captured["overrides"], {})
        self.assertNotIn("with_simulation", captured["overrides"])
        self.assertNotIn("REPORT|PREFLIGHT|OK", captured["body"])
        check_names = [c["name"] for c in result["verification"]["checks"]]
        self.assertNotIn("report_marker", check_names)

    def test_with_simulation_defaults_to_preflight(self):
        result, captured = self._run(with_simulation=True)
        self.assertEqual(
            captured["overrides"]["with_simulation"],
            {"current_A": 1.0, "freq_hz": 3000.0, "run_solve": False},
        )
        self.assertIn("REPORT|PREFLIGHT|OK", captured["body"])
        self.assertNotIn("RunSimulation", captured["body"])
        self.assertEqual(result["status"], "ok", msg=str(result["verification"]))
        marker = [c for c in result["verification"]["checks"]
                  if c["name"] == "report_marker"]
        self.assertEqual(len(marker), 1)
        self.assertTrue(marker[0]["pass"], msg=marker[0]["detail"])
        self.assertIn("PREFLIGHT", marker[0]["detail"])

    def test_with_simulation_run_solve_marker(self):
        result, captured = self._run(with_simulation=True, run_solve=True,
                                     current_A=2500.0, freq_hz=2500.0)
        sim = captured["overrides"]["with_simulation"]
        self.assertTrue(sim["run_solve"])
        self.assertEqual(sim["current_A"], 2500.0)
        self.assertEqual(sim["freq_hz"], 2500.0)
        self.assertIn("REPORT|SOLVE|HasResults|", captured["body"])
        marker = [c for c in result["verification"]["checks"]
                  if c["name"] == "report_marker"]
        self.assertEqual(len(marker), 1)
        self.assertTrue(marker[0]["pass"], msg=marker[0]["detail"])
        self.assertIn("SOLVE", marker[0]["detail"])

    def test_report_marker_fails_when_absent(self):
        # 预检标记未出现在 stdout 时验证应失败
        with _Harness(self, self._tmp.name) as h:
            def _run_no_marker(script_path, timeout_s=600):
                Path(h._smash).touch()
                return {"exit_code": 0, "stdout": "REPORT|DONE\n",
                        "stderr": "", "duration_s": 0.1}
            with patch.object(s4l_model.s4l_script, "_s4l_run_script", _run_no_marker):
                result = s4l_model._s4l_model(radius=0.05, with_simulation=True)
        self.assertEqual(result["status"], "failed_verification")
        marker = [c for c in result["verification"]["checks"]
                  if c["name"] == "report_marker"]
        self.assertEqual(len(marker), 1)
        self.assertFalse(marker[0]["pass"])

    def test_head_model_default_layers_override_true(self):
        result, captured = self._run(with_head_model=True)
        self.assertIs(captured["overrides"]["with_head_model"], True)
        self.assertEqual(result["status"], "ok", msg=str(result["verification"]))
        self.assertIn("scalp", captured["expected"]["entity_names"])

    def test_head_model_custom_layers(self):
        layers = [["brain", 0.07, "db", 1], ["scalp", 0.085, "db", 3]]
        result, captured = self._run(with_head_model=True, head_layers=layers)
        self.assertEqual(captured["overrides"]["with_head_model"],
                         {"layers": layers})
        self.assertEqual(result["status"], "ok", msg=str(result["verification"]))

    def test_sim_name_passthrough_only_when_non_default(self):
        _, captured = self._run(with_simulation=True)
        self.assertNotIn("sim_name", captured["overrides"])

        result, captured = self._run(with_simulation=True, sim_name="sim_custom")
        self.assertEqual(captured["overrides"]["sim_name"], "sim_custom")
        self.assertIn("sim_custom", captured["body"])
        self.assertEqual(result["status"], "ok", msg=str(result["verification"]))


if __name__ == "__main__":
    unittest.main()
