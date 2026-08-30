#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gui_wrap 单元测试（默认集，无 S4L 依赖）。

覆盖：
- 三件套内容断言：XCore.GetApp() 自适应段、buffering=1、os._exit(0)；
- log_path 正确嵌入（含 Windows 反斜杠路径不炸转义）；
- body 逐字节嵌入；
- 确定性：同一输入两次调用产物逐字节相同；
- py_compile 守卫：包装平凡 body 的产物可编译。
"""

from __future__ import annotations

import os
import py_compile
import tempfile
import unittest

from research_agent.s4lmodel.gui_wrap import emit_gui_script

BODY = (
    'import s4l_v1.document as document\n'
    'document.New()\n'
    'print("REPORT|ENTITY_COUNT|0")\n'
    'print("REPORT|DONE")\n'
)


class ThreePieceTests(unittest.TestCase):
    def setUp(self):
        self.out = emit_gui_script(BODY, "D:/artifacts/run.log")

    def test_context_adaptive_header(self):
        self.assertIn("import XCore", self.out)
        self.assertIn("XCore.GetApp()", self.out)
        self.assertIn("run_application()", self.out)

    def test_log_file_line_buffered(self):
        self.assertIn('open(r"D:/artifacts/run.log", "w", buffering=1)', self.out)
        # stdout/stderr 均重定向，REPORT 落日志文件
        self.assertIn("sys.stdout = _log", self.out)
        self.assertIn("sys.stderr = _log", self.out)

    def test_os_exit(self):
        self.assertIn("os._exit(0)", self.out)

    def test_no_real_s4l_import_in_wrapper(self):
        # 包装器自身不得 import s4l_v1（AST 级检查，忽略字符串字面量）
        import ast
        import inspect
        from research_agent.s4lmodel import gui_wrap

        tree = ast.parse(inspect.getsource(gui_wrap))
        names = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                names.extend(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                names.append(n.module or "")
        self.assertFalse([m for m in names if m.startswith("s4l_v1")],
                         f"gui_wrap.py 不得真实 import s4l_v1: {names}")


class LogPathEmbeddingTests(unittest.TestCase):
    def test_windows_backslash_path_no_escape_corruption(self):
        # 反斜杠路径含 \t、\r 段：必须转正斜杠嵌入，不能炸转义
        out = emit_gui_script(BODY, "D:\\TMS\\results\\task1\\run.log")
        self.assertIn('open(r"D:/TMS/results/task1/run.log", "w", buffering=1)', out)
        self.assertNotIn("\\TMS", out)

    def test_forward_slash_path_kept(self):
        out = emit_gui_script(BODY, "D:/artifacts/x.log")
        self.assertIn('r"D:/artifacts/x.log"', out)


class BodyEmbeddingTests(unittest.TestCase):
    def test_body_byte_exact(self):
        out = emit_gui_script(BODY, "D:/a.log")
        self.assertIn(BODY, out)


class DeterminismTests(unittest.TestCase):
    def test_same_input_byte_identical(self):
        a = emit_gui_script(BODY, "D:\\x\\y.log")
        b = emit_gui_script(BODY, "D:\\x\\y.log")
        self.assertEqual(a, b)


class PyCompileGuardTests(unittest.TestCase):
    def test_wrapped_trivial_body_compiles(self):
        out = emit_gui_script('print("REPORT|DONE")\n', "D:\\t\\r.log")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(out)
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
