#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""figure8_synthesis 与 tms_optimize_compiler 映射/边界测试。"""

from __future__ import annotations

import unittest

from research_agent.literature.param_model import FieldValue, ParamValue, SimulationParams
from research_agent.literature.synthesis import get_compiler, get_synthesis
from research_agent.literature.synthesis.tms_figure8 import figure8_synthesis
from research_agent.literature.synthesis.tms_optimize_compiler import tms_optimize_compiler


def _fv(value, unit=None):
    return FieldValue(value=ParamValue(value=value, unit=unit), quote="q")


def _params(**fields):
    return SimulationParams(template="tms_figure8", fields=fields)


class Figure8SynthesisTests(unittest.TestCase):
    def test_wing_diameter_to_coil_sphere_radius(self):
        p = _params(wing_diameter=_fv(0.1, "m"))
        task = figure8_synthesis(p)
        self.assertEqual(task.geometry_intent["kind"], "coil_sphere")
        self.assertAlmostEqual(task.geometry_intent["params"]["radius"], 0.05)

    def test_focal_depth_to_field_target(self):
        p = _params(wing_diameter=_fv(0.1, "m"), focal_depth_m=_fv(0.02, "m"))
        task = figure8_synthesis(p)
        self.assertEqual(task.field_target["depth_m"], 0.02)

    def test_wire_diameter_m_to_mm(self):
        p = _params(wire_diameter=_fv(0.0053, "m"))
        task = figure8_synthesis(p)
        self.assertAlmostEqual(task.constraints["wire_diameter_mm"], 5.3)

    def test_i_peak_to_constraints(self):
        p = _params(I_peak=_fv(5000, "A"))
        task = figure8_synthesis(p)
        self.assertEqual(task.constraints["max_current_A"], 5000.0)

    def test_approximations_recorded(self):
        p = _params(wing_diameter=_fv(0.1, "m"), focal_depth_m=_fv(0.02, "m"))
        task = figure8_synthesis(p)
        self.assertTrue(len(task.meta["approximations"]) >= 2)

    def test_missing_fields_defaults(self):
        task = figure8_synthesis(_params())
        self.assertEqual(task.constraints["pulse_rise_time_us"], 100.0)
        self.assertEqual(task.geometry_intent["kind"], "wing_pair")


class TmsOptimizeCompilerTests(unittest.TestCase):
    def test_radius_and_depth(self):
        from research_agent.literature.synthesis import BackendTask

        task = BackendTask(
            geometry_intent={"kind": "coil_sphere", "params": {"radius": 0.05}},
            field_target={"depth_m": 0.02, "region_ellipses": [{"a": 0.2}]},
            constraints={"wire_diameter_mm": 5.3, "max_current_A": 5000},
        )
        cfg = tms_optimize_compiler(task, {})
        self.assertAlmostEqual(cfg["coil_radius"], 0.05)
        self.assertAlmostEqual(cfg["head_radius"], 0.03)
        self.assertEqual(cfg["manufacturability"]["wire_diameter_mm"], 5.3)
        self.assertEqual(cfg["manufacturability"]["max_current_A"], 5000.0)

    def test_depth_exceeds_radius_clamps(self):
        from research_agent.literature.synthesis import BackendTask

        task = BackendTask(
            geometry_intent={"kind": "coil_sphere", "params": {"radius": 0.01}},
            field_target={"depth_m": 0.05},
        )
        cfg = tms_optimize_compiler(task, {})
        self.assertEqual(cfg["head_radius"], 0.001)
        self.assertIn("_warn", cfg)

    def test_no_depth_head_equals_coil(self):
        from research_agent.literature.synthesis import BackendTask

        task = BackendTask(geometry_intent={"kind": "coil_sphere", "params": {"radius": 0.07}})
        cfg = tms_optimize_compiler(task, {})
        self.assertAlmostEqual(cfg["head_radius"], 0.07)

    def test_registry(self):
        self.assertIsNotNone(get_synthesis("tms_figure8"))
        self.assertIsNotNone(get_compiler("tms_optimize"))


if __name__ == "__main__":
    unittest.main()
