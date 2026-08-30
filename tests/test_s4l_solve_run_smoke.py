#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_solve_run 分级冒烟（真实 S4L 调用，skipUnless 分级，不进 CI 默认套件）。

- PreflightSmoke（headless，免 license）：s4l_model 预检模式全链
  （编译 → headless 干跑 → REPORT|PREFLIGHT|OK 断言），小几何单环；
- SolveRunSmoke（GUI 全链，QS_SOLVER 单座）：_s4l_solve_run B4 配方
  （双翼 9 匝、翼径 0.1m、线径 5.3mm、三层头模）→ 分 stage 报告 →
  Output.h5 存在 + field_metrics 关键键 + 验证判定。

license 单座：SolveRunSmoke 严格串行，不得与其他 S4L 任务并发。
GUI 冷启动 3-5min + 求解数分钟，timeout 充裕。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from research_agent.config import SETTINGS

_HEADLESS_READY = (
    bool(SETTINGS.s4l_python)
    and Path(SETTINGS.s4l_python).is_file()
    and bool(SETTINGS.s4l_home)
    and Path(SETTINGS.s4l_home).is_dir()
)

_GUI_READY = bool(SETTINGS.s4l_gui) and Path(SETTINGS.s4l_gui).is_file()


@unittest.skipUnless(_HEADLESS_READY, "需要 Sim4Life headless 环境（S4L_PYTHON/S4L_HOME）")
class PreflightSmoke(unittest.TestCase):
    """headless 预检：s4l_model 预检模式（run_solve=False，免 license）。"""

    def test_preflight_single_loop_with_head_model(self):
        from research_agent.tools import s4l_model

        r = s4l_model._s4l_model(
            radius=0.05, turns_per_wing=1, wire_diameter_mm=2.0,
            with_simulation=True, with_head_model=True, run_solve=False,
        )
        self.assertEqual(r["status"], "ok", f"预检冒烟失败：{r}")
        checks = {c["name"]: c["pass"] for c in r["verification"]["checks"]}
        self.assertTrue(checks.get("report_marker"),
                        f"缺 REPORT|PREFLIGHT|OK：{r['verification']}")


@unittest.skipUnless(_GUI_READY, "需要 S4L GUI 执行器（S4L_GUI，QS_SOLVER 单座）")
class SolveRunSmoke(unittest.TestCase):
    """GUI 全链：_s4l_solve_run B4 配方真实求解 + 场验证。"""

    def test_b4_recipe_full_chain(self):
        from research_agent.tools import s4l_solve_run

        r = s4l_solve_run._s4l_solve_run(
            wing_diameter=0.1, turns_per_wing=9, wire_diameter_mm=5.3,
            with_head_model=True, timeout_s=2400,
        )
        self.assertIn(r["status"], ("ok", "failed_verification"),
                      f"全链冒烟失败：stage={r.get('stage')} error={r.get('error')}")
        # 各 stage 均落账 duration
        for name in ("compile", "preflight", "wrap", "run", "locate",
                     "extract", "verify"):
            self.assertIn(name, r["stages"], f"缺 stage {name}：{r['stages'].keys()}")
            self.assertTrue(r["stages"][name]["ok"], f"stage {name} 失败：{r}")
            self.assertIn("duration_s", r["stages"][name])
        self.assertTrue(Path(r["output_h5"]).is_file(),
                        f"Output.h5 不存在：{r['output_h5']}")
        for key in ("peak_depth_m", "peak_E", "E_per_I_at_depths",
                    "single_peak", "monotonic_decay"):
            self.assertIn(key, r["field_metrics"], f"field_metrics 缺 {key}")
        self.assertIn(r["verification"]["verdict"], ("pass", "fail"))


if __name__ == "__main__":
    unittest.main()
