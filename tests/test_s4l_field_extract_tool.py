#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_field_extract 工具单测：mock 提取/判定层，零 S4L 依赖。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_agent.s4lmodel import field_extract
from research_agent.tools import s4l_field_extract


class LocateTests(unittest.TestCase):
    def test_no_args_is_locate_error(self):
        r = s4l_field_extract._s4l_field_extract()
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "locate")

    def test_empty_results_dir_is_locate_error(self):
        with tempfile.TemporaryDirectory() as td:
            r = s4l_field_extract._s4l_field_extract(results_dir=td)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "locate")

    def test_missing_h5_is_locate_error(self):
        r = s4l_field_extract._s4l_field_extract(h5_path="D:/nonexistent/x_Output.h5")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "locate")

    def test_results_dir_uses_latest_output(self):
        with tempfile.TemporaryDirectory() as td:
            old = Path(td) / "a_Output.h5"
            new = Path(td) / "b_Output.h5"
            old.write_bytes(b"x")
            new.write_bytes(b"x")
            with (
                patch.object(field_extract, "extract_focal_profile",
                             return_value={"peak_E": 1.0}) as m_ext,
                patch.object(field_extract, "verify_field_features",
                             return_value={"verdict": "pass", "checks": [], "notes": []}),
            ):
                r = s4l_field_extract._s4l_field_extract(results_dir=td)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["h5_path"], str(new))
        m_ext.assert_called_once_with(str(new), current_A=1.0)


class ExtractTests(unittest.TestCase):
    def test_h5_direct_ok(self):
        with tempfile.TemporaryDirectory() as td:
            h5 = Path(td) / "x_Output.h5"
            h5.write_bytes(b"x")
            metrics = {"peak_E": 6e-2, "peak_depth_m": 0.0}
            verdict = {"verdict": "pass", "checks": [], "notes": []}
            with (
                patch.object(field_extract, "extract_focal_profile",
                             return_value=metrics) as m_ext,
                patch.object(field_extract, "verify_field_features",
                             return_value=verdict) as m_ver,
            ):
                r = s4l_field_extract._s4l_field_extract(
                    h5_path=str(h5), current_A=2.0, criteria={"peak_E_range": (0, 1)})
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["metrics"], metrics)
        self.assertEqual(r["verification"], verdict)
        m_ext.assert_called_once_with(str(h5), current_A=2.0)
        m_ver.assert_called_once_with(metrics, {"peak_E_range": (0, 1)})

    def test_extract_exception_is_extract_error(self):
        with tempfile.TemporaryDirectory() as td:
            h5 = Path(td) / "x_Output.h5"
            h5.write_bytes(b"x")
            with patch.object(field_extract, "extract_focal_profile",
                              side_effect=KeyError("no AllFields")):
                r = s4l_field_extract._s4l_field_extract(h5_path=str(h5))
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "extract")
        self.assertEqual(r["h5_path"], str(h5))


if __name__ == "__main__":
    unittest.main()
