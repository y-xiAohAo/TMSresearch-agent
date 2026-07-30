#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""内省管线结构断言测试：词汇表 JSON 的结构完整性。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parents[1] / "src" / "research_agent" / "literature" / "vocab"


class PrimitivesStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prims = json.loads((VOCAB_DIR / "primitives.json").read_text(encoding="utf-8"))

    def test_min_pyaedt_primitives(self):
        pyaedt = {k: v for k, v in self.prims.items() if not k.startswith("s4l_")}
        self.assertGreaterEqual(len(pyaedt), 10, "至少应导出 10 个 PyAEDT create_* 签名")

    def test_core_primitives_present(self):
        for name in ("create_cylinder", "create_box", "create_sphere", "create_polyline"):
            self.assertIn(name, self.prims, f"缺核心原语 {name}")

    def test_param_fields_structure(self):
        for name, spec in self.prims.items():
            self.assertIn("params", spec, f"{name} 缺 params")
            self.assertIn("source", spec, f"{name} 缺 source")
            for p in spec["params"]:
                self.assertIn("name", p)
                self.assertIn("type", p)
                self.assertIn("required", p)

    def test_enum_extraction(self):
        cyl_orient = [p for p in self.prims["create_cylinder"]["params"] if p["name"] == "orientation"][0]
        self.assertEqual(cyl_orient["type"], "enum", "orientation 应为 enum 类型")

    def test_polyline_segment_enum_options(self):
        seg = [p for p in self.prims["create_polyline"]["params"] if p["name"] == "segment_type"]
        if seg:
            opts = seg[0].get("enum_options", [])
            for expected in ("Line", "Arc", "Spline"):
                self.assertIn(expected, opts, f"segment_type 枚举缺 {expected}")


class DeviceTemplateStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = json.loads((VOCAB_DIR / "device_templates.json").read_text(encoding="utf-8"))

    def test_templates_present(self):
        for name in ("dipole", "patch", "tms_figure8"):
            self.assertIn(name, self.templates, f"缺设备模板 {name}")

    def test_template_fields_have_units(self):
        for tname, spec in self.templates.items():
            self.assertIn("fields", spec)
            self.assertIn("source", spec)
            for f in spec["fields"]:
                self.assertIn("name", f)
                self.assertIn("type", f)

    def test_dipole_key_fields(self):
        names = [f["name"] for f in self.templates["dipole"]["fields"]]
        for key in ("dipole_length", "frequency"):
            self.assertIn(key, names, f"dipole 模板缺关键字段 {key}")


if __name__ == "__main__":
    unittest.main()
