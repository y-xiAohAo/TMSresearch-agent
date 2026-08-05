#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S4lHeadSmoke：B4 头模编译产物 headless 体素化验证（无需 license）。

链路：compiler(with_head_model+with_simulation) → headless 执行（建模/网格/
体素化/WriteInputFile 均不需要 license；最后的 RunSimulation 会报 QS_SOLVER——
预期行为，Input.h5 已在此前落盘）→ h5py 断言三层组织体素计数。
"""

from __future__ import annotations

import glob
import os
import unittest
from pathlib import Path

from research_agent.config import SETTINGS


def _s4l_ready() -> bool:
    return bool(SETTINGS.s4l_python) and Path(SETTINGS.s4l_python).is_file()


@unittest.skipUnless(_s4l_ready(), "需要 Sim4Life headless 环境")
class S4lHeadSmoke(unittest.TestCase):
    def test_head_model_voxelization(self):
        import h5py
        import numpy as np
        from research_agent.literature.synthesis import BackendTask, get_compiler
        from research_agent.tools import s4l_script

        smash = str(Path(SETTINGS.artifacts_dir) / "head_smoke.smash")
        task = BackendTask(
            geometry_intent={"kind": "coil_sphere", "params": {
                "radius": 0.05, "wing_diameter": 0.1, "turns_per_wing": 2}},
            constraints={"wire_diameter_mm": 2.0},
        )
        out = get_compiler("sim4life")(task, {
            "smash_path": smash,
            "with_head_model": True,
            "with_simulation": {},
        })
        script = s4l_script._s4l_write_script(
            out["script_body"], filename="s4l_head_smoke.py")["script_path"]
        # headless：求解段会失败（QS_SOLVER），属预期；Input.h5 已落盘
        s4l_script._s4l_run_script(script, timeout_s=600)

        inputs = glob.glob(smash + "_Results" + os.sep + "*_Input.h5")
        self.assertTrue(inputs, f"Input.h5 未生成：{smash}_Results")
        with h5py.File(sorted(inputs)[-1], "r") as h5:
            mesh = list(h5["Meshes"].values())[0]
            v = np.array(mesh["voxels"])
            nm = dict(mesh["name_map"].attrs)
        names = {i: nm.get(f"_{i}", b"").decode() for i in range(int(nm.get("_num_entries", 0)))}
        counts = {names.get(int(i), str(i)): int((v == i).sum()) for i in np.unique(v)}
        # 三层组织均非零，且量级关系 brain > skull > scalp（体积决定）
        for layer in ("brain", "skull", "scalp"):
            self.assertGreater(counts.get(layer, 0), 500,
                               f"{layer} 体素过少：{counts}")
        self.assertGreater(counts["brain"], counts["skull"])
        self.assertGreater(counts["skull"], counts["scalp"])


if __name__ == "__main__":
    unittest.main()
