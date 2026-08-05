#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B4 单元 2.5 单元测试：头模发射器 + compiler with_head_model（默认集，无 S4L 依赖）。"""

from __future__ import annotations

import unittest

from research_agent.literature.synthesis import BackendTask, get_compiler
from research_agent.s4lmodel import emlf_setup, head_geometry


def _wing_task() -> BackendTask:
    return BackendTask(
        geometry_intent={"kind": "coil_sphere", "params": {
            "radius": 0.05, "wing_diameter": 0.1, "turns_per_wing": 2,
        }},
        constraints={"wire_diameter_mm": 2.0},
    )


class HeadGeometryTests(unittest.TestCase):
    def test_default_layers_and_naming(self):
        body, names = head_geometry.emit_head_shells()
        self.assertEqual(names, ["scalp", "skull", "brain"])  # 大→小创建
        self.assertEqual(body.count("model.CreateSolidSphere("), 3)
        self.assertIn("0.092", body)  # scalp 半径
        # 头模置于线圈下方
        body2, _ = head_geometry.emit_head_shells(center=(0, 0, -0.094))
        self.assertIn("-0.094", body2)

    def test_material_pairs_and_voxeler_layers(self):
        pairs = head_geometry.head_material_pairs()
        self.assertIn(("brain", "Brain (Grey Matter)"), pairs)
        self.assertIn(("skull", "Skull Cortical"), pairs)
        vox = dict(head_geometry.head_voxeler_layers())
        self.assertGreater(vox["brain"], vox["skull"])
        self.assertGreater(vox["skull"], vox["scalp"])  # 内层高优先级


class MaterialLinkEmitterTests(unittest.TestCase):
    def test_link_chain_content(self):
        body = emlf_setup.emit_material_links([("brain", "Brain (Grey Matter)")])
        self.assertIn("s4l_v1.materials", body)  # 数据库模块正确路径
        self.assertIn("FindMaterial", body)
        self.assertIn("AddMaterialSettings", body)
        self.assertIn("LinkMaterialWithDatabase", body)
        self.assertIn("REPORT|MATLINK|", body)  # σ 读回核验
        self.assertNotIn("entity.Material =", body)  # pitfall #23 禁路


class LossyGridVoxelerTests(unittest.TestCase):
    def test_grid_binding_and_maxstep(self):
        body = emlf_setup.emit_mqs_simulation(
            "sim", ["w1"], 1.0, 0.001,
            grid_entities=["w1", "brain"], max_step_m=0.002, padding_m=0.05,
        )
        # 网格绑定实体列表（pitfall #39 无参=不纳入）
        self.assertIn("AddAutomaticGridSettings(_ents_by_names(", body)
        # MaxStep/padding 嵌套元组（pitfall #22）
        self.assertIn("((0.002, 0.002, 0.002), units.Meters)", body)
        self.assertIn("ManualPadding = True", body)
        self.assertIn("BottomPadding = ((0.05, 0.05, 0.05), units.Meters)", body)

    def test_default_grid_unchanged(self):
        # 不传 grid_entities 时保持 B3 形态（回归）
        body = emlf_setup.emit_mqs_simulation("sim", ["w1"], 1.0, 0.001)
        self.assertIn("sim.AddAutomaticGridSettings()", body)

    def test_layered_voxeler_solve(self):
        body = emlf_setup.emit_solve(voxeler_layers=[("brain", 30), ("skull", 20)])
        self.assertIn("AddAutomaticVoxelerSettings(_ents_by_names(['brain']))", body)
        self.assertIn("_avs.Priority = 30", body)
        self.assertIn("kIntersectionVoxeler", body)  # pitfall #39
        self.assertIn("run_isolve_directly=True", body)
        # 顺序铁律
        pos = [body.index(s) for s in
               (".UpdateGrid()", "AddAutomaticVoxelerSettings", ".CreateVoxels()",
                "AllSimulations.Add", "RunSimulation")]
        self.assertEqual(pos, sorted(pos))

    def test_default_solve_unchanged(self):
        body = emlf_setup.emit_solve()
        self.assertIn("sim.AddAutomaticVoxelerSettings()", body)
        self.assertNotIn("kIntersectionVoxeler", body)


class CompilerHeadModelTests(unittest.TestCase):
    SMASH = "D:/fake/test_b4.smash"

    def _compile(self, overrides):
        o = {"smash_path": self.SMASH}
        o.update(overrides)
        return get_compiler("sim4life")(_wing_task(), o)

    def test_head_model_geometry(self):
        out = self._compile({"with_head_model": True})
        body = out["script_body"]
        self.assertEqual(body.count("model.CreateSolidSphere("), 3)
        self.assertIn("brain", out["expected"]["entity_names"])
        # 头模替代空气域球；头模实体不走字符串材料指派
        self.assertNotIn("air", out["expected"]["entity_names"])
        self.assertNotIn("brain", out["expected"]["materials"])
        # 默认 SetLengthUnits（pitfall #38）
        self.assertIn("SetLengthUnits(units.Meters)", body)

    def test_head_plus_simulation_full_recipe(self):
        out = self._compile({"with_head_model": True, "with_simulation": {}})
        body = out["script_body"]
        # 全配方：网格绑定 + MaxStep + 材料链接 + 分层体素器 + Intersection
        self.assertIn("AddAutomaticGridSettings(_ents_by_names(", body)
        self.assertIn("units.Meters)", body)
        self.assertIn("REPORT|MATLINK|", body)
        self.assertIn("_avs.Priority = 30", body)
        self.assertIn("kIntersectionVoxeler", body)
        # 顺序：材料链接在求解段之前
        self.assertLess(body.index("REPORT|MATLINK|"), body.index(".RunSimulation("))
        notes = " ".join(out["notes"])
        self.assertIn("头模近似", notes)

    def test_simulation_without_head_unchanged(self):
        # 回归：with_simulation 但无头模 → 无分层体素器/材料链接
        out = self._compile({"with_simulation": {}})
        body = out["script_body"]
        self.assertNotIn("MATLINK", body)
        self.assertNotIn("kIntersectionVoxeler", body)
        self.assertIn("sim.AddAutomaticGridSettings()", body)

    def test_compiled_body_is_valid_python(self):
        import py_compile, tempfile, os
        out = self._compile({"with_head_model": True, "with_simulation": {}})
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(out["script_body"])
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
