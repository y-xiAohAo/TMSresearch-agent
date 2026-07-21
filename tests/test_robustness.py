#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""鲁棒性测试：故障路径行为——不幻觉、不静默、可诊断。

- RAG 服务宕机：sim4life_manual_qa 明确报错
- TMS 超时：返回 status=timeout 且结构完整
- s4l 脚本缺失/语法错误：exit_code != 0 且 stderr 回传
- tracking 钩子：工具调用后 JSONL 有记录
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.tracking import track_tool


class RagDownRobustnessTests(unittest.TestCase):
    def test_rag_down_raises_clear_error_not_hallucination(self):
        from research_agent.tools import sim4life_manual_qa

        with patch.object(
            sim4life_manual_qa.requests, "get", side_effect=ConnectionError("refused")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                sim4life_manual_qa._sim4life_manual_qa("创建项目步骤？")
        msg = str(ctx.exception)
        self.assertIn("RAG 服务不可达", msg)
        self.assertIn("uvicorn", msg)  # 给出可操作的恢复指引


class TmsTimeoutRobustnessTests(unittest.TestCase):
    def test_tms_timeout_returns_structured_partial(self):
        from research_agent.tools import tms_optimize
        import subprocess

        with patch.object(
            tms_optimize.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd="tms", timeout=1),
        ):
            with patch.object(tms_optimize.Path, "is_file", return_value=True):
                result = tms_optimize._tms_optimize({"Nsteps": 2}, budget_s=1)
        self.assertEqual(result["status"], "timeout")
        self.assertIn("config_path", result)
        self.assertIn("duration_s", result)
        self.assertIsInstance(result["metrics"], dict)


class S4lFailureRobustnessTests(unittest.TestCase):
    def test_missing_script_raises_file_not_found(self):
        from research_agent.tools import s4l_script
        from research_agent.config import SETTINGS

        if not Path(SETTINGS.s4l_python).is_file():
            self.skipTest("需要本机 Sim4Life 安装")
        with self.assertRaises(FileNotFoundError):
            s4l_script._s4l_run_script("D:/nonexistent/ghost.py")

    def test_syntax_error_script_returns_nonzero_with_stderr(self):
        from research_agent.tools import s4l_script
        from research_agent.config import SETTINGS
        import tempfile

        if not Path(SETTINGS.s4l_python).is_file():
            self.skipTest("需要本机 Sim4Life 安装")
        bad = Path(tempfile.mkdtemp(prefix="s4l_bad_")) / "bad.py"
        bad.write_text("def broken(:\n    pass\n", encoding="utf-8")
        result = s4l_script._s4l_run_script(str(bad), timeout_s=60)
        self.assertNotEqual(result["exit_code"], 0)
        self.assertTrue(result["stderr"])


class TrackingHookTests(unittest.TestCase):
    def test_track_tool_writes_jsonl_record(self):
        import tempfile
        import research_agent.tracking as tracking

        log_file = Path(tempfile.mkdtemp(prefix="track_")) / "experiment_log.jsonl"
        with patch.object(tracking, "_log_path", return_value=log_file):
            @track_tool
            def fake_tool(x: int) -> dict:
                return {"v": x * 2}

            fake_tool(x=3)

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["tool"], "fake_tool")
        self.assertTrue(record["success"])
        self.assertIn("duration_s", record)

    def test_track_tool_records_failure(self):
        import tempfile
        import research_agent.tracking as tracking

        log_file = Path(tempfile.mkdtemp(prefix="track_")) / "experiment_log.jsonl"
        with patch.object(tracking, "_log_path", return_value=log_file):
            @track_tool
            def failing_tool() -> dict:
                raise ValueError("boom")

            with self.assertRaises(ValueError):
                failing_tool()

        record = json.loads(log_file.read_text(encoding="utf-8").strip())
        self.assertFalse(record["success"])
        self.assertIn("boom", record["error"])


if __name__ == "__main__":
    unittest.main()
