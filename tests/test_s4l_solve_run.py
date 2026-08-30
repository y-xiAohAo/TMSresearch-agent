#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_solve_run 编排单测：mock 子进程层与提取/对比层，零 S4L 依赖。

覆盖：happy path 全绿（stages duration 齐）、预检失败不触 GUI（spec D5）、
GUI 超时 → stage=run、找不到 Output.h5 → stage=locate、ref 定量锚路径、
shape mismatch 降级特征模式、磁盘保有（5 个假输出只留最新 3 个）。
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from research_agent.s4lmodel import field_extract, h5compare
from research_agent.tools import s4l_script, s4l_solve, s4l_solve_run

PASS_VERDICT = {"verdict": "pass", "checks": [], "notes": []}
FAKE_METRICS = {"peak_E": 6e-2, "peak_depth_m": 0.0}
PREFLIGHT_OK_RUN = {"stdout": "...\nREPORT|PREFLIGHT|OK\n", "stderr": "",
                    "exit_code": 0, "duration_s": 1.0, "artifacts": []}
GUI_OK_RUN = {"exit_code": 0, "timed_out": False,
              "log": "REPORT|ENTITY_COUNT|5\nREPORT|SOLVE|HasResults|True\nREPORT|DONE\n",
              "duration_s": 420.0}


def _fake_compiler(smash_path: str):
    """假 compiler：预检/求解两版 body 各带对应 REPORT 标记。"""
    def compiler(task, overrides=None):
        overrides = overrides or {}
        run_solve = overrides.get("with_simulation", {}).get("run_solve", True)
        marker = "REPORT|SOLVE|HasResults|" if run_solve else "REPORT|PREFLIGHT|OK"
        return {
            "script_body": f"# fake body run_solve={run_solve}\nprint(\"{marker}\")\n",
            "expected": {"smash_path": smash_path, "entity_count_min": 2,
                         "entity_names": [], "materials": {}},
            "notes": ["fake"],
        }
    return compiler


def _patch_layers(test, smash_path="D:/tmp_fake/model.smash",
                  preflight_run=None, gui_run=None, output_h5="D:/tmp_fake/model.smash_Results/x_Output.h5"):
    """统一 mock：compiler / headless 预检 / GUI 执行 / 定位 / 提取 / 判定。"""
    m_gui = MagicMock(return_value=gui_run if gui_run is not None else GUI_OK_RUN)
    patches = [
        patch.object(s4l_solve_run, "get_compiler",
                     return_value=_fake_compiler(smash_path)),
        patch.object(s4l_script, "_s4l_run_script",
                     return_value=preflight_run if preflight_run is not None else PREFLIGHT_OK_RUN),
        patch.object(s4l_solve, "_s4l_run_gui", m_gui),
        patch.object(field_extract, "find_latest_output_h5",
                     return_value=output_h5),
        patch.object(field_extract, "extract_focal_profile",
                     return_value=FAKE_METRICS),
        patch.object(field_extract, "verify_field_features",
                     return_value=PASS_VERDICT),
    ]
    for p in patches:
        p.start()
        test.addCleanup(p.stop)
    return m_gui


class HappyPathTests(unittest.TestCase):
    def test_happy_path_all_stages(self):
        m_gui = _patch_layers(self)
        r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07, turns_per_wing=9,
                                         current_A=1.0, freq_hz=3000.0)
        self.assertEqual(r["status"], "ok", msg=str(r)[:500])
        for stage in ("compile", "preflight", "wrap", "run",
                      "locate", "extract", "verify"):
            self.assertIn(stage, r["stages"])
            self.assertIn("duration_s", r["stages"][stage])
            self.assertTrue(r["stages"][stage]["ok"])
        self.assertEqual(r["verification"]["verify_mode"], "feature")
        self.assertEqual(r["output_h5"],
                         "D:/tmp_fake/model.smash_Results/x_Output.h5")
        self.assertIn("artifacts_size_mb", r)
        m_gui.assert_called_once()
        # GUI 脚本是 gui_wrap 产物（XCore 自适应头），未过 headless 引导头
        script_text = Path(r["script_path"]).read_text(encoding="utf-8")
        self.assertIn("XCore.GetApp()", script_text)
        self.assertNotIn("run_application()\n\n_app", script_text[:200])

    def test_geometry_input_validation(self):
        r = s4l_solve_run._s4l_solve_run()  # wing_diameter 与 radius 都缺
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "compile")

    def test_real_compiler_integration(self):
        """真 compiler（纯函数）走通 compile 段，预检/求解 body 标记各就各位。"""
        captured = {}

        def fake_write(body, filename="x.py"):
            captured[filename] = body
            return {"script_path": "D:/tmp_fake/x.py"}

        with (
            patch.object(s4l_script, "_s4l_write_script", side_effect=fake_write),
            patch.object(s4l_script, "_s4l_run_script", return_value=PREFLIGHT_OK_RUN),
            patch.object(s4l_solve, "_s4l_run_gui", return_value=GUI_OK_RUN),
            patch.object(field_extract, "find_latest_output_h5",
                         return_value="D:/tmp_fake/x_Output.h5"),
            patch.object(field_extract, "extract_focal_profile",
                         return_value=FAKE_METRICS),
            patch.object(field_extract, "verify_field_features",
                         return_value=PASS_VERDICT),
            patch.object(s4l_solve_run.gui_wrap, "emit_gui_script",
                         side_effect=lambda body, log: body),
        ):
            r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07, turns_per_wing=9)
        self.assertEqual(r["status"], "ok", msg=str(r)[:500])
        self.assertIn("REPORT|PREFLIGHT|OK", captured["s4l_preflight.py"])
        self.assertTrue(r["smash_path"].endswith(".smash"))
        self.assertEqual(r["stages"]["run"]["duration_gui_s"], 420.0)


class StageFailureTests(unittest.TestCase):
    def test_preflight_failure_never_touches_gui(self):
        """spec D5 证据用例：预检失败 → stage=preflight 且 _s4l_run_gui 未被调用。"""
        bad = dict(PREFLIGHT_OK_RUN, exit_code=1, stdout="", stderr="boom")
        m_gui = _patch_layers(self, preflight_run=bad)
        r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "preflight")
        m_gui.assert_not_called()

    def test_preflight_missing_marker_fails(self):
        bad = dict(PREFLIGHT_OK_RUN, stdout="no marker here")
        m_gui = _patch_layers(self, preflight_run=bad)
        r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07)
        self.assertEqual(r["stage"], "preflight")
        m_gui.assert_not_called()

    def test_gui_timeout_is_run_error(self):
        timeout_run = {"exit_code": -9, "timed_out": True,
                       "log": "partial log tail", "duration_s": 1800.0}
        _patch_layers(self, gui_run=timeout_run)
        r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07, timeout_s=1800)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "run")
        self.assertIn("log_tail", r)

    def test_missing_output_h5_is_locate_error(self):
        _patch_layers(self, output_h5=None)
        r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "locate")

    def test_no_has_results_in_log_is_locate_error(self):
        bad_gui = dict(GUI_OK_RUN, log="REPORT|DONE\n")
        _patch_layers(self, gui_run=bad_gui)
        r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "locate")


class VerifyTests(unittest.TestCase):
    def test_ref_h5_reference_mode_pass(self):
        _patch_layers(self)
        stats = {"fields": {}, "worst": {"max_rel_err": 0.0, "p99_rel_err": 0.0}}
        with patch.object(h5compare, "compare_fields", return_value=stats) as m_cmp:
            r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07, ref_h5="D:/ref.h5")
        self.assertEqual(r["status"], "ok")
        v = r["verification"]
        self.assertEqual(v["verify_mode"], "reference")
        self.assertEqual(v["verdict"], "pass")
        m_cmp.assert_called_once_with(r["output_h5"], "D:/ref.h5")

    def test_ref_h5_reference_mode_fail(self):
        _patch_layers(self)
        stats = {"fields": {}, "worst": {"max_rel_err": 0.05, "p99_rel_err": 0.02}}
        with patch.object(h5compare, "compare_fields", return_value=stats):
            r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07, ref_h5="D:/ref.h5")
        self.assertEqual(r["status"], "failed_verification")
        self.assertEqual(r["verification"]["verdict"], "fail")

    def test_shape_mismatch_falls_back_to_feature(self):
        _patch_layers(self)
        with patch.object(h5compare, "compare_fields",
                          side_effect=ValueError("shape mismatch")):
            r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07, ref_h5="D:/ref.h5")
        v = r["verification"]
        self.assertEqual(v["verify_mode"], "feature_fallback_shape_mismatch")
        self.assertIn("fallback_reason", v)
        self.assertEqual(v["verdict"], "pass")  # 特征判据结果被保留
        self.assertEqual(r["status"], "ok")


class RetentionTests(unittest.TestCase):
    def test_keeps_latest_3_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            smash = str(Path(td) / "model.smash")
            Path(smash).write_bytes(b"x")
            results_dir = Path(smash + "_Results")
            results_dir.mkdir()
            base = time.time() - 1000
            names = []
            for i in range(5):
                p = results_dir / f"run{i}_Output.h5"
                p.write_bytes(b"x" * 100)
                os.utime(p, (base + i * 10, base + i * 10))
                names.append(str(p))

            m_gui = MagicMock(return_value=GUI_OK_RUN)
            with (
                patch.object(s4l_solve_run, "get_compiler",
                             return_value=_fake_compiler(smash)),
                patch.object(s4l_script, "_s4l_run_script",
                             return_value=PREFLIGHT_OK_RUN),
                patch.object(s4l_solve, "_s4l_run_gui", m_gui),
                patch.object(field_extract, "extract_focal_profile",
                             return_value=FAKE_METRICS),
                patch.object(field_extract, "verify_field_features",
                             return_value=PASS_VERDICT),
            ):
                # find_latest_output_h5 用真的：应选中 mtime 最新的 run4
                r = s4l_solve_run._s4l_solve_run(wing_diameter=0.07)

            self.assertEqual(r["status"], "ok", msg=str(r)[:500])
            self.assertTrue(r["output_h5"].endswith("run4_Output.h5"))
            remaining = sorted(p.name for p in results_dir.glob("*_Output.h5"))
            self.assertEqual(remaining,
                             ["run2_Output.h5", "run3_Output.h5", "run4_Output.h5"])
            self.assertEqual(sorted(Path(c).name for c in r["cleaned"]),
                             ["run0_Output.h5", "run1_Output.h5"])
            self.assertEqual(len(r["outputs_kept"]), 3)
            self.assertAlmostEqual(r["artifacts_size_mb"], 0.0, places=1)


if __name__ == "__main__":
    unittest.main()
