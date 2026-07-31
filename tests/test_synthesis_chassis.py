#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""synthesis 底盘测试：BackendTask 结构 + 两个注册表。"""

from __future__ import annotations

import unittest

from research_agent.literature.synthesis import (
    BackendTask,
    get_compiler,
    get_synthesis,
    list_compilers,
    list_syntheses,
    register_compiler,
    register_synthesis,
)


class BackendTaskStructureTests(unittest.TestCase):
    def test_defaults(self):
        task = BackendTask()
        self.assertEqual(task.geometry_intent, {})
        self.assertEqual(task.field_target, {})
        self.assertEqual(task.constraints, {})
        self.assertEqual(task.outputs, [])
        self.assertEqual(task.meta, {})

    def test_fields(self):
        task = BackendTask(
            geometry_intent={"kind": "coil_sphere", "params": {"radius": 0.05}},
            field_target={"depth_m": 0.02},
            constraints={"max_current_A": 5000},
            outputs=["focality"],
            meta={"approximations": ["翼形≈球面"]},
        )
        self.assertEqual(task.geometry_intent["kind"], "coil_sphere")
        self.assertIn("翼形≈球面", task.meta["approximations"])


class RegistryTests(unittest.TestCase):
    def test_synthesis_register_and_get(self):
        @register_synthesis("_test_dev")
        def _fn(params):
            return BackendTask()

        self.assertIsNotNone(get_synthesis("_test_dev"))
        self.assertIn("_test_dev", list_syntheses())
        self.assertIsNone(get_synthesis("_nonexistent"))

    def test_compiler_register_and_get(self):
        @register_compiler("_test_backend")
        def _fn(task, overrides):
            return {"compiled": True}

        self.assertIsNotNone(get_compiler("_test_backend"))
        self.assertIn("_test_backend", list_compilers())
        self.assertIsNone(get_compiler("_nonexistent"))

    def test_chassis_extensible_without_modification(self):
        """通用性证明：可注册新 synthesis/compiler 而不改底盘代码。"""
        @register_synthesis("_ext_dev")
        def _new(params):
            return BackendTask(geometry_intent={"kind": "wing_pair"})

        @register_compiler("_ext_backend")
        def _newc(task, overrides):
            return {"ok": 1}

        self.assertIsNotNone(get_synthesis("_ext_dev"))
        self.assertIsNotNone(get_compiler("_ext_backend"))
        # 直接可用
        task = get_synthesis("_ext_dev")(None)
        self.assertEqual(task.geometry_intent["kind"], "wing_pair")
        self.assertEqual(get_compiler("_ext_backend")(task, {}), {"ok": 1})


if __name__ == "__main__":
    unittest.main()
