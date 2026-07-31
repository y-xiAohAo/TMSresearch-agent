#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S4lModelSmoke：B2 冒烟测试——真实 S4L headless 执行（默认排除）。

排除方式：pytest -k "not S4lModelSmoke"（与 S4lScriptSmoke 同级）。
需要 S4L_PYTHON / S4L_HOME 配置且真实存在；无 S4L 主机自动 skip。

每个用例内核启动 60-100s，串行执行。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from research_agent.config import SETTINGS
from research_agent.tools.s4l_model import _s4l_model

_S4L_READY = bool(SETTINGS.s4l_python) and Path(SETTINGS.s4l_python).is_file() \
    and bool(SETTINGS.s4l_home) and Path(SETTINGS.s4l_home).is_dir()


@unittest.skipUnless(_S4L_READY, "Sim4Life 未安装/未配置")
class S4lModelSmokeTests(unittest.TestCase):
    """B2a：单环线圈最小链路（模板→脚本→headless→.smash）。"""

    def test_b2a_single_loop(self):
        result = _s4l_model(radius=0.05, wire_diameter_mm=2.0, timeout_s=600)
        self.assertEqual(result["status"], "ok",
                         f"验证失败：{result.get('structured_error') or result['verification']}")
        v = result["verification"]
        self.assertTrue(v["all_pass"])
        self.assertTrue(Path(result["smash_path"]).is_file())
        # 2 基线 + 1 环 + 1 空气球
        self.assertGreaterEqual(v["report"]["entity_count"], 4)


@unittest.skipUnless(_S4L_READY, "Sim4Life 未安装/未配置")
class S4lModelSmokeFigure8Tests(unittest.TestCase):
    """B2b：figure8 双翼多匝（对接 B1 抽取的真实论文参数 2511.00744）。"""

    def test_b2b_figure8_wing_pair(self):
        result = _s4l_model(
            wing_diameter=0.1, turns_per_wing=9, wire_diameter_mm=5.3,
            timeout_s=1500,  # 实测 19 实体 + 材料指派 >600s
        )
        self.assertEqual(result["status"], "ok",
                         f"验证失败：{result.get('structured_error') or result['verification']}")
        v = result["verification"]
        self.assertTrue(v["all_pass"])
        # 2 基线 + 18 匝 + 1 空气球
        self.assertGreaterEqual(v["report"]["entity_count"], 21)
        names = v["report"]["entity_names"]
        self.assertIn("wing_l_turn0", names)
        self.assertIn("wing_r_turn8", names)
        self.assertIn("air", names)


if __name__ == "__main__":
    unittest.main()
