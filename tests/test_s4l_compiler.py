#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_compiler 单元测试（默认集，无 S4L 依赖）。

覆盖 spec §5：
- sim4life compiler 已注册；
- BackendTask → script_body 参数注入正确；
- script_body 不含引导头、含 SaveAs 与实体报告打印；
- 组件层独立性：figure8_geometry 不 import setup 以外的仿真设置逻辑；
- expected 契约字段齐全。
"""

from __future__ import annotations

import unittest

from research_agent.literature.synthesis import BackendTask, get_compiler
from research_agent.s4lmodel import figure8_geometry, setup


def _wing_task() -> BackendTask:
    return BackendTask(
        geometry_intent={"kind": "coil_sphere", "params": {
            "radius": 0.05, "wing_diameter": 0.1, "turns_per_wing": 9,
        }},
        constraints={"wire_diameter_mm": 5.3, "max_current_A": 5300},
        meta={"approximations": ["近似1: 翼形半径≈球面半径"]},
    )


class CompilerRegistrationTests(unittest.TestCase):
    def test_sim4life_registered(self):
        self.assertIsNotNone(get_compiler("sim4life"))


class ScriptBodyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = get_compiler("sim4life")(_wing_task(), {})
        cls.body = cls.out["script_body"]

    def test_contract_fields(self):
        self.assertIn("script_body", self.out)
        self.assertIn("expected", self.out)
        exp = self.out["expected"]
        for key in ("entity_count_min", "entity_names", "smash_path"):
            self.assertIn(key, exp)
        self.assertTrue(exp["smash_path"].endswith(".smash"))

    def test_no_bootstrap_header(self):
        # 引导头由 s4l_write_script 加，模板不得自带
        self.assertNotIn("run_application", self.body)

    def test_param_injection(self):
        # wing_diameter=0.1 → 翼半径 0.05；wire 5.3mm → 0.0053m；turns=9
        self.assertIn("0.05", self.body)
        self.assertIn("0.0053", self.body)
        self.assertEqual(self.body.count("CreateSolidTube"), 18)  # 2 翼 × 9 匝
        self.assertIn("wing_l_turn8", self.body)
        self.assertIn("wing_r_turn0", self.body)

    def test_setup_components(self):
        self.assertIn("document.New()", self.body)
        self.assertIn("CreateSolidSphere", self.body)  # 空气域
        self.assertIn("SaveAs", self.body)
        self.assertIn("REPORT|ENTITY_COUNT|", self.body)
        self.assertIn("MaterialName", self.body)  # 材料指派

    def test_materials_in_expected(self):
        mats = self.out["expected"]["materials"]
        self.assertEqual(mats["wing_l_turn0"], "Copper")
        self.assertEqual(mats["air"], "Air")
        self.assertEqual(len(mats), 19)  # 18 匝 + 空气球

    def test_expected_entity_count(self):
        # 2 基线 + 18 匝 + 1 空气球
        self.assertEqual(self.out["expected"]["entity_count_min"], 21)
        self.assertEqual(len(self.out["expected"]["entity_names"]), 19)

    def test_notes_carry_approximations(self):
        notes = " ".join(self.out["notes"])
        self.assertIn("近似1", notes)
        self.assertIn("同心圆环组", notes)


class SingleLoopTests(unittest.TestCase):
    def test_radius_only_falls_back_to_single_loop(self):
        task = BackendTask(
            geometry_intent={"kind": "coil_sphere", "params": {"radius": 0.03}},
            constraints={"wire_diameter_mm": 2.0},
        )
        out = get_compiler("sim4life")(task, {})
        self.assertEqual(out["script_body"].count("CreateSolidTube"), 1)
        self.assertIn("loop_turn0", out["expected"]["entity_names"])

    def test_overrides(self):
        out = get_compiler("sim4life")(_wing_task(), {"air_radius": 0.5})
        self.assertIn("0.5", out["script_body"])


class GeometryEmitterTests(unittest.TestCase):
    def test_clamp_when_turns_too_many(self):
        body, names = figure8_geometry.emit_wing_pair(
            wing_diameter=0.02, turns_per_wing=100, wire_diameter=0.005,
        )
        self.assertIn("clamped", body)
        self.assertEqual(len(names), 200)

    def test_component_layer_independence(self):
        # 架构约束：几何模块不得 import setup（拓扑特化不碰共享组件）
        import inspect
        src = inspect.getsource(figure8_geometry)
        self.assertNotIn("import setup", src)
        self.assertNotIn("from research_agent.s4lmodel.setup", src)

    def test_setup_emitters_pure(self):
        # 共享组件为纯发射器：无真实 s4l_v1 import（AST 级检查，忽略字符串字面量）
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(setup))
        imports = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        names = []
        for n in imports:
            if isinstance(n, ast.Import):
                names.extend(a.name for a in n.names)
            else:
                names.append(n.module or "")
        self.assertFalse([m for m in names if m.startswith("s4l_v1")],
                         f"setup.py 不得真实 import s4l_v1: {names}")


class VerifyParsingTests(unittest.TestCase):
    def test_parse_report(self):
        from research_agent.tools.s4l_model import _parse_report, _verify

        stdout = ("noise\nREPORT|ENTITY_COUNT|21\nREPORT|ENTITY|wing_l_turn0\n"
                  "REPORT|MATERIAL|wing_l_turn0|Copper\nREPORT|DONE\n")
        rep = _parse_report(stdout)
        self.assertEqual(rep["entity_count"], 21)
        self.assertIn("wing_l_turn0", rep["entity_names"])
        self.assertEqual(rep["materials"]["wing_l_turn0"], "Copper")
        self.assertTrue(rep["report_done"])

    def test_verify_pass_and_fail(self):
        from research_agent.tools.s4l_model import _verify

        run_ok = {"exit_code": 0,
                  "stdout": "REPORT|ENTITY_COUNT|3\nREPORT|ENTITY|a\nREPORT|DONE\n"}
        exp = {"entity_count_min": 3, "entity_names": ["a"],
               "smash_path": __file__}  # 存在的文件即可
        self.assertTrue(_verify(run_ok, exp)["all_pass"])

        run_bad = {"exit_code": 1, "stdout": ""}
        v = _verify(run_bad, exp)
        self.assertFalse(v["all_pass"])


class WithSimulationTests(unittest.TestCase):
    """B3 仿真级扩展：with_simulation overrides（spec §9.2）。"""

    SMASH = "D:/fake/test_b3.smash"

    def _compile(self, overrides):
        return get_compiler("sim4life")(_wing_task(), overrides)

    def _expected_b2_body(self):
        """直接组合 B2 发射器得到的基线 body（回归保护基准）。"""
        geom_body, names = figure8_geometry.emit_wing_pair(
            wing_diameter=0.1, turns_per_wing=9, wire_diameter=0.0053,
        )
        mats = {n: "Copper" for n in names if n != "air"}
        mats["air"] = "Air"
        return "\n".join([
            setup.emit_header(),
            geom_body,
            setup.emit_air_domain((0.0, 0.0, 0.0), 0.2),
            setup.emit_assign_material(mats),
            setup.emit_save_and_report(self.SMASH),
        ])

    def test_b2_path_byte_identical_without_sim_cfg(self):
        # 回归保护：不开 with_simulation 时与 B2 已验证路径逐字节一致
        out = self._compile({"smash_path": self.SMASH})
        self.assertEqual(out["script_body"], self._expected_b2_body())
        self.assertNotIn("CreateCircle", out["script_body"])
        self.assertNotIn("CreateVoxels", out["script_body"])

    def test_with_simulation_segments(self):
        out = self._compile({"smash_path": self.SMASH,
                             "with_simulation": {"current_A": 2.0}})
        body = out["script_body"]
        # 电流源线框：每匝一个 CreateCircle（pitfall #17）
        self.assertEqual(body.count("model.CreateCircle("), 18)
        self.assertIn("wing_l_turn0_wire", out["expected"]["entity_names"])
        self.assertEqual(out["expected"]["entity_count_min"], 21 + 18)
        # MQS 仿真段：类型化绑定 + 单位元组 + 两翼反向
        self.assertIn("sim.AddCurrentSourceSettings(", body)
        self.assertIn("(2, units.Amperes)", body)
        self.assertIn("_cs_neg.IsDirectionReverted = True", body)
        # 求解顺序铁律：SaveAs 在 CreateVoxels 前（pitfall #20）
        self.assertLess(body.index("document.SaveAs("), body.index(".CreateVoxels()"))
        self.assertIn(".RunSimulation(wait=True, run_isolve_directly=True)", body)
        # B2 几何段原样保留
        geom_body, _ = figure8_geometry.emit_wing_pair(0.1, 9, 0.0053)
        self.assertIn(geom_body, body)
        # 诚实声明进 notes
        notes = " ".join(out["notes"])
        self.assertIn("nothing to solve", notes)
        self.assertIn("IsDirectionReverted", notes)

    def test_with_simulation_body_is_valid_python(self):
        import py_compile, tempfile, os
        out = self._compile({"smash_path": self.SMASH, "with_simulation": {}})
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(out["script_body"])
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
        finally:
            os.unlink(path)

    def test_single_loop_sim_has_no_negative_group(self):
        task = BackendTask(
            geometry_intent={"kind": "coil_sphere", "params": {"radius": 0.03}},
            constraints={"wire_diameter_mm": 2.0},
        )
        out = get_compiler("sim4life")(task, {
            "smash_path": self.SMASH, "with_simulation": {}})
        body = out["script_body"]
        self.assertIn("loop_turn0_wire", out["expected"]["entity_names"])
        self.assertNotIn("coil_neg", body)  # 单环无负组，不发射空绑定
        self.assertEqual(body.count("model.CreateCircle("), 1)


if __name__ == "__main__":
    unittest.main()
