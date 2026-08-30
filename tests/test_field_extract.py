#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""field_extract 模块测试：合成 h5 fixture + 真实存量 b4 数据标定回归。"""

from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

import sys
sys.path.insert(0, "src")

from research_agent.s4lmodel import field_extract  # noqa: E402

REAL_H5 = ("artifacts/b4_head_figure8.smash_Results/"
           "6a2d4140-2880-48c5-8390-38fef2fcd6c6_Output.h5")


def _write_synthetic_h5(path: str) -> dict:
    """造一个小型 Output.h5：|E| = E0 * exp(-depth/λ) 沿 z 自表面单调衰减。

    结构模仿 h5compare 的读取路径；comp0 在 x 维 node-centered（交错），
    其余分量 cell-centered，验证 _to_cell_centers 插值。z>0 全 NaN（域外），
    表面在 z=0。
    """
    e0, lam = 0.05, 0.020
    ax_x = np.linspace(-0.02, 0.02, 9)     # 8 cells
    ax_y = np.linspace(-0.02, 0.02, 9)
    ax_z = np.linspace(-0.05, 0.01, 31)    # 30 cells, 2mm
    zc = 0.5 * (ax_z[:-1] + ax_z[1:])
    # cell-centered |E| 目标场（含 NaN 域外）
    depth = -zc.clip(max=0.0)              # 表面 z=0 以下深度
    field = e0 * np.exp(-depth / lam)
    field[zc > 0.0] = np.nan
    f3 = np.broadcast_to(field[None, None, :], (8, 8, 30)).copy()

    with h5py.File(path, "w") as h5:
        grp = h5.create_group("FieldGroups/aaaaaaaa/AllFields/"
                              "EM E(x,y,z,f0)/_Object/Snapshots/0")
        # comp0：x 维 node-centered（shape 9,8,30），存 |E| 于 Ey 不方便，
        # 直接把全部幅值放 comp1（cell-centered），comp0/comp2 置零
        comp0 = np.zeros((9, 8, 30, 2))
        comp1 = np.zeros((8, 8, 30, 2))
        comp1[..., 0] = f3
        comp2 = np.zeros((8, 8, 30, 2))
        grp.create_dataset("comp0", data=comp0)
        grp.create_dataset("comp1", data=comp1)
        grp.create_dataset("comp2", data=comp2)
        mesh = h5.create_group("Meshes/bbbbbbbb")
        mesh.create_dataset("axis_x", data=ax_x)
        mesh.create_dataset("axis_y", data=ax_y)
        mesh.create_dataset("axis_z", data=ax_z)
    return {"e0": e0, "lam": lam}


class TestFindLatestOutputH5(unittest.TestCase):
    def test_picks_latest_by_mtime(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.path.join(d, "111_Output.h5")
            new = os.path.join(d, "222_Output.h5")
            other = os.path.join(d, "333_Output.txt")
            for p in (old, new, other):
                open(p, "w").close()
            os.utime(old, (1_000_000, 1_000_000))
            os.utime(new, (2_000_000, 2_000_000))
            os.utime(other, (3_000_000, 3_000_000))  # 非 h5 不应被选中
            self.assertEqual(field_extract.find_latest_output_h5(d), new)

    def test_empty_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(field_extract.find_latest_output_h5(d))


class TestExtractFocalProfileSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.h5_path = os.path.join(cls._tmp.name, "synth_Output.h5")
        cls.params = _write_synthetic_h5(cls.h5_path)
        cls.metrics = field_extract.extract_focal_profile(cls.h5_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_peak_value_and_depth(self):
        m = self.metrics
        # 峰值在表面第一层 cell-center（z=-1mm 处，E = e0*exp(-0.001/λ)≈0.95*e0）
        self.assertAlmostEqual(m["peak_E"], self.params["e0"] * np.exp(-0.001 / 0.02),
                               delta=1e-3)
        self.assertAlmostEqual(m["peak_depth_m"], 0.0, places=3)

    def test_depth_profile_matches_exp_decay(self):
        m = self.metrics
        lam = self.params["lam"]
        for d, v in m["E_per_I_at_depths"].items():
            d = float(d)
            expected = self.params["e0"] * np.exp(-d / lam)
            self.assertAlmostEqual(v, expected, delta=2e-3,
                                   msg=f"depth {d}")

    def test_single_peak_and_monotonic(self):
        m = self.metrics
        self.assertTrue(m["single_peak"])
        self.assertTrue(m["monotonic_decay"])
        self.assertEqual(m["decay_inversions"], 0)

    def test_verify_passes_with_default_criteria(self):
        v = field_extract.verify_field_features(self.metrics)
        self.assertEqual(v["verdict"], "pass",
                         msg=str(v["checks"]))

    def test_current_scaling(self):
        m = field_extract.extract_focal_profile(self.h5_path, current_A=100.0)
        self.assertAlmostEqual(m["peak_E"], self.metrics["peak_E"] / 100.0,
                               places=12)


@unittest.skipUnless(os.path.exists(REAL_H5), "真实 b4 存量数据不在本机")
class TestRealB4Data(unittest.TestCase):
    """对 275MB 真实 Output.h5 的标定回归（b4 spec 记录值）。"""

    @classmethod
    def setUpClass(cls):
        cls.metrics = field_extract.extract_focal_profile(REAL_H5)

    def test_verify_passes(self):
        v = field_extract.verify_field_features(self.metrics)
        self.assertEqual(v["verdict"], "pass", msg=str(v["checks"]))

    def test_peak_matches_spec(self):
        # b4 spec：头表峰值 E/I = 6.01e-2 V/m/A
        self.assertAlmostEqual(self.metrics["peak_E"], 6.01e-2, delta=6.01e-3)

    def test_depth_15_25mm_within_10pct_of_spec(self):
        # b4 spec：15–25mm 深度 ≈1.5–2e-2 V/m/A，允许 10% 偏差
        e = {float(k): v for k, v in self.metrics["E_per_I_at_depths"].items()}
        self.assertAlmostEqual(e[0.020], 1.78e-2, delta=1.78e-3)
        for d in (0.015, 0.020, 0.025):
            self.assertTrue(1.35e-2 <= e[d] <= 2.75e-2,
                            msg=f"depth {d}: {e[d]:.3e}")

    def test_peak_at_head_surface(self):
        self.assertLessEqual(self.metrics["peak_depth_m"], 0.005)


if __name__ == "__main__":
    unittest.main()
