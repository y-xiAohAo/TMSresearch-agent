#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3 单元测试：解析解 + 对比判定逻辑（默认集，无 S4L 依赖）。"""

from __future__ import annotations

import unittest

from research_agent.s4lmodel import analytical


class AnalyticalTests(unittest.TestCase):
    def test_center_field(self):
        # z=0：B = μ₀I/(2R)。R=0.05, I=1A → 4πe-7/(0.1) ≈ 1.2566e-5 T
        b = analytical.loop_axial_field(0.05, 1.0, 0.0)
        self.assertAlmostEqual(b, analytical.MU0 / 0.1, places=15)

    def test_far_field_cubic_decay(self):
        # z>>R 时 B ∝ 1/z³：z 翻倍场约降 8 倍
        b1 = analytical.loop_axial_field(0.01, 1.0, 1.0)
        b2 = analytical.loop_axial_field(0.01, 1.0, 2.0)
        self.assertAlmostEqual(b1 / b2, 8.0, delta=0.05)

    def test_linearity_in_current(self):
        b1 = analytical.loop_axial_field(0.05, 1.0, 0.05)
        b2 = analytical.loop_axial_field(0.05, 2.0, 0.05)
        self.assertAlmostEqual(b2 / b1, 2.0, places=12)

    def test_concentric_sum(self):
        radii = [0.05, 0.045, 0.04]
        total = analytical.concentric_rings_axial_field(radii, 1.0, 0.02)
        manual = sum(analytical.loop_axial_field(r, 1.0, 0.02) for r in radii)
        self.assertAlmostEqual(total, manual, places=18)
        self.assertGreater(total, analytical.loop_axial_field(0.05, 1.0, 0.02))

    def test_wing_radii_matches_geometry_rule(self):
        # 与 figure8_geometry 同规则：wing_r - k*wire_d，clamp 到 2*wire_r
        radii = analytical.wing_radii(0.1, 3, 0.0053)
        self.assertEqual(len(radii), 3)
        self.assertAlmostEqual(radii[0], 0.05)
        self.assertAlmostEqual(radii[1], 0.05 - 0.0053)
        # clamp 情形
        clamped = analytical.wing_radii(0.02, 10, 0.005)
        self.assertTrue(all(r >= 0.005 for r in clamped))


class VerdictTests(unittest.TestCase):
    """对比判定三档逻辑（B3a ±5%，B3b 20%/50%）。"""

    def _verdict(self, rel_errs, pass_tol, partial_tol=None):
        from research_agent.tools.s4l_solve import _judge
        return _judge(rel_errs, pass_tol, partial_tol)

    def test_b3a_pass_boundary(self):
        self.assertEqual(self._verdict([0.049, 0.01], 0.05), "pass")
        self.assertNotEqual(self._verdict([0.051], 0.05), "pass")

    def test_b3b_three_bands(self):
        self.assertEqual(self._verdict([0.15], 0.20, 0.50), "pass")
        self.assertEqual(self._verdict([0.30], 0.20, 0.50), "partial")
        self.assertEqual(self._verdict([0.60], 0.20, 0.50), "fail")


class EmlfSetupEmitterTests(unittest.TestCase):
    """emlf_setup 定稿发射器内容断言（已验证调用序列，spec §9.2）。"""

    def test_wires_use_create_circle_and_naming(self):
        from research_agent.s4lmodel import emlf_setup
        body, names = emlf_setup.emit_current_source_wires([
            {"name": "wing_l_turn0", "center": (-0.05, 0.0, 0.0), "radius": 0.05},
            {"name": "wing_r_turn0", "center": (0.05, 0.0, 0.0), "radius": 0.04},
        ])
        self.assertEqual(names, ["wing_l_turn0_wire", "wing_r_turn0_wire"])
        self.assertEqual(body.count("model.CreateCircle("), 2)
        self.assertIn('Vec3(-0.05, 0, 0)', body)
        self.assertIn('Vec3(0, 0, 1)', body)  # 默认法向 +z
        self.assertIn('_w.Name = "wing_l_turn0_wire"', body)

    def test_mqs_simulation_verified_api(self):
        from research_agent.s4lmodel import emlf_setup
        body = emlf_setup.emit_mqs_simulation(
            "sim_tms", ["wing_l_turn0_wire"], 1.0, 0.001,
            negative_wires=["wing_r_turn0_wire"],
        )
        # 类型化绑定 + 单位元组（裸浮点求解器解释不同）
        self.assertIn("sim.AddCurrentSourceSettings(", body)
        self.assertIn("units.Amperes", body)
        self.assertIn("units.Meters", body)
        # 两翼反向（figure8 物理关键）
        self.assertIn("_cs_neg.IsDirectionReverted = True", body)
        self.assertIn("_cs_pos.IsDirectionReverted = False", body)
        # 网格/传感器/频率/材料更新
        self.assertIn("sim.AddAutomaticGridSettings()", body)
        self.assertIn("_sensor.RecordHField = True", body)
        self.assertIn("sim.SetupSettings.Frequency = 3000", body)
        self.assertIn("sim.UpdateAllMaterials()", body)
        # 已废弃/已否定 API 不得出现
        self.assertNotIn("AddToUI", body)
        self.assertNotIn("MagneticBFieldEvaluator", body)
        self.assertNotIn("sim.Add(_cs", body)

    def test_solve_order_iron_law(self):
        from research_agent.s4lmodel import emlf_setup
        body = emlf_setup.emit_solve()
        order = [
            ".UpdateGrid()",
            ".AddAutomaticVoxelerSettings()",
            ".CreateVoxels()",
            "document.AllSimulations.Add(sim)",
            ".RunSimulation(wait=True, run_isolve_directly=True)",
        ]
        pos = [body.index(s) for s in order]
        self.assertEqual(pos, sorted(pos), "求解顺序铁律被破坏")
        self.assertNotIn("document.SaveAs(", body)  # SaveAs 调用归上游 emit_save_and_report

    def test_emitted_body_is_valid_python(self):
        # pitfall #33：生成脚本必须 py_compile 检查——发射层先保证语法有效
        import py_compile, tempfile, os
        from research_agent.s4lmodel import emlf_setup, setup
        wires_body, names = emlf_setup.emit_current_source_wires([
            {"name": "loop_turn0", "center": (0, 0, 0), "radius": 0.05},
        ])
        body = (setup.emit_header() + wires_body
                + emlf_setup.emit_mqs_simulation("sim", names, 1.0, 0.001)
                + setup.emit_save_and_report("dummy.smash")
                + emlf_setup.emit_solve())
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(body)
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
        finally:
            os.unlink(path)


class H5CompareTests(unittest.TestCase):
    """h5compare 合成 fixture 单测（结构仿真实 Output.h5）。"""

    FIELD = "EM E(x,y,z,f0)"

    def _write_h5(self, path, comps):
        """comps: {"comp0": (nx,ny,nz) 实数数组} → 写成 (...,2) (re,im) 布局。"""
        import h5py
        import numpy as np
        with h5py.File(path, "w") as h5:
            snaps = (h5.create_group("FieldGroups/u1/AllFields")
                     .create_group(f"{self.FIELD}/_Object/Snapshots/0"))
            for name, arr in comps.items():
                arr = np.asarray(arr, dtype=np.float64)
                d = np.zeros(arr.shape + (2,), dtype=np.float64)
                d[..., 0] = arr
                snaps.create_dataset(name, data=d)

    def _base_comps(self):
        import numpy as np
        rng = np.random.default_rng(7)
        base = rng.uniform(0.1, 10.0, size=(6, 7, 8))
        base[0, 0, 0] = np.nan  # 域外体素
        return {"comp0": base, "comp1": base * 1.5, "comp2": base * 0.5}

    def test_identical_is_zero_err(self):
        import tempfile, os
        from research_agent.s4lmodel import h5compare
        comps = self._base_comps()
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.h5")
            b = os.path.join(td, "b.h5")
            self._write_h5(a, comps)
            self._write_h5(b, {k: v.copy() for k, v in comps.items()})
            r = h5compare.compare_fields(a, b, (self.FIELD,))
        self.assertEqual(r["worst"]["max_rel_err"], 0.0)
        self.assertEqual(r["fields"][self.FIELD]["comp0"]["n_voxels"],
                         6 * 7 * 8 - 1)  # NaN 体素被排除

    def test_shape_mismatch_raises(self):
        import tempfile, os
        import numpy as np
        from research_agent.s4lmodel import h5compare
        comps = self._base_comps()
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.h5")
            b = os.path.join(td, "b.h5")
            self._write_h5(a, comps)
            bad = {k: np.zeros((3, 3, 3)) for k in comps}
            self._write_h5(b, bad)
            with self.assertRaises(ValueError):
                h5compare.compare_fields(a, b, (self.FIELD,))

    def test_perturbed_voxel_stats(self):
        import tempfile, os
        from research_agent.s4lmodel import h5compare
        comps = self._base_comps()
        sim = {k: v.copy() for k, v in comps.items()}
        sim["comp0"][3, 3, 3] *= 1.5  # 单强场体素 +50%
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.h5")
            b = os.path.join(td, "b.h5")
            self._write_h5(a, sim)
            self._write_h5(b, comps)
            r = h5compare.compare_fields(a, b, (self.FIELD,))
        c0 = r["fields"][self.FIELD]["comp0"]
        self.assertAlmostEqual(c0["max_rel_err"], 0.5, places=6)
        self.assertAlmostEqual(c0["median_rel_err"], 0.0, places=6)

    def test_weak_field_masked(self):
        import tempfile, os
        from research_agent.s4lmodel import h5compare
        comps = self._base_comps()
        # 构造一个确定的弱场体素（<< 1% nanmax），sim 侧扰动 100 倍也不应进入统计
        comps["comp0"][1, 1, 1] = 1e-6
        sim = {k: v.copy() for k, v in comps.items()}
        sim["comp0"][1, 1, 1] = 1e-4
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.h5")
            b = os.path.join(td, "b.h5")
            self._write_h5(a, sim)
            self._write_h5(b, comps)
            r = h5compare.compare_fields(a, b, (self.FIELD,))
        self.assertEqual(r["worst"]["max_rel_err"], 0.0)

    def test_linearity_scale2(self):
        import tempfile, os
        from research_agent.s4lmodel import h5compare
        comps = self._base_comps()
        sim = {k: v * 2.0 for k, v in comps.items()}  # NaN 传播保持
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "a.h5")
            b = os.path.join(td, "b.h5")
            self._write_h5(a, sim)
            self._write_h5(b, comps)
            r = h5compare.compare_linearity(a, b, 2.0, (self.FIELD,))
        for comp in ("comp0", "comp1", "comp2"):
            s = r["fields"][self.FIELD][comp]
            self.assertAlmostEqual(s["ratio_median"], 2.0, places=6)
            self.assertAlmostEqual(s["ratio_rel_err_vs_scale"], 0.0, places=6)


class GuiRerunScriptTests(unittest.TestCase):
    """_emit_gui_rerun_script 内容断言（GUI --run 三件套 + 探针纪律）。"""

    def test_gui_three_piece_set(self):
        from research_agent.tools.s4l_solve import _emit_gui_rerun_script
        body = _emit_gui_rerun_script("D:/x/bm.smash", "R1", "D:/x/run.log")
        self.assertIn("XCore.GetApp()", body)          # 上下文自适应
        self.assertNotIn("run_application", body)       # GUI 内禁止（pitfall #7）
        self.assertIn('buffering=1', body)              # 日志写文件（pitfall #8）
        self.assertIn("os._exit(0)", body)              # 退出（pitfall #9）
        self.assertIn('document.Open(r"D:/x/bm.smash")', body)
        self.assertIn("document.AllSimulations", body)  # 属性非方法（pitfall #21）
        self.assertIn("RunSimulation(wait=True, run_isolve_directly=True)", body)  # Ares 笼统错误规避
        self.assertNotIn("RUN|SCALE|", body)            # 无扰动块

    def test_amplitude_scale_block(self):
        from research_agent.tools.s4l_solve import _emit_gui_rerun_script
        body = _emit_gui_rerun_script("D:/x/bm.smash", "R1", "D:/x/run.log",
                                      amplitude_scale=2.0)
        self.assertIn("RUN|SCALE|count|", body)
        self.assertIn("_v * 2.0", body)
        self.assertIn("AmplitudeProp", body)  # 探针定案：GUI 写路径走 Property 对象
        self.assertIn("RUN|SCALE|verify|", body)

    def test_backup_block_single_file(self):
        from research_agent.tools.s4l_solve import _emit_gui_rerun_script
        body = _emit_gui_rerun_script("D:/x/bm.smash", "R1", "D:/x/run.log",
                                      backup_dir="D:/x/bak")
        # 单文件备份（磁盘纪律）：GetOutputFileName + copy2，不全量复制目录
        self.assertIn("GetOutputFileName", body)
        self.assertIn("RUN|BACKUP|ok", body)
        self.assertIn("shutil", body)
        # 备份块必须在求解之前（防覆盖）
        self.assertLess(body.index("RUN|BACKUP|ok"), body.index("RunSimulation"))
        body_no = _emit_gui_rerun_script("D:/x/bm.smash", "R1", "D:/x/run.log")
        self.assertNotIn("RUN|BACKUP|", body_no)

    def test_script_is_valid_python(self):
        import py_compile, tempfile, os
        from research_agent.tools.s4l_solve import _emit_gui_rerun_script
        body = _emit_gui_rerun_script("D:/x/bm.smash", "R1", "D:/x/run.log", 2.0)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(body)
            path = f.name
        try:
            py_compile.compile(path, doraise=True)
        finally:
            os.unlink(path)


class OutputSnapshotTests(unittest.TestCase):
    """复算输出定位（mtime 快照差分，pitfall #29）。"""

    def test_snapshot_and_pick(self):
        import os, tempfile, time
        from research_agent.tools.s4l_solve import _snapshot_outputs, _pick_recomputed
        with tempfile.TemporaryDirectory() as td:
            a = os.path.join(td, "u1_Output.h5")
            b = os.path.join(td, "u2_Output.h5")
            for p in (a, b):
                with open(p, "w") as f:
                    f.write("x")
            before = _snapshot_outputs(td)
            self.assertEqual(set(before), {a, b})
            # 未变化 → None
            self.assertIsNone(_pick_recomputed(before, _snapshot_outputs(td)))
            # b 被重写（mtime 前进）→ 唯一变化者
            time.sleep(0.02)
            with open(b, "a") as f:
                f.write("y")
            os.utime(b, (time.time() + 5, time.time() + 5))
            self.assertEqual(_pick_recomputed(before, _snapshot_outputs(td)), b)

    def test_parse_scale_count(self):
        from research_agent.tools.s4l_solve import _parse_scale_count
        self.assertEqual(_parse_scale_count("RUN|SIM|R1\nRUN|SCALE|count|3\n"), 3)
        self.assertEqual(_parse_scale_count("RUN|SCALE|count|0\n"), 0)
        self.assertIsNone(_parse_scale_count("RUN|DONE\n"))

    def test_parse_scale_verify(self):
        from research_agent.tools.s4l_solve import _parse_scale_verify
        log = ("RUN|SIM|R1\n"
               "RUN|SCALE|verify|10000.0\n"
               "RUN|SCALE|Wire Current Settings|5000.0\n"
               "RUN|SCALE|count|1\n")
        self.assertEqual(_parse_scale_verify(log), [(5000.0, 10000.0)])
        # no-op 情形：读回=原值
        bad = ("RUN|SCALE|verify|5000.0\n"
               "RUN|SCALE|Wire Current Settings|5000.0\n")
        self.assertEqual(_parse_scale_verify(bad), [(5000.0, 5000.0)])
        self.assertEqual(_parse_scale_verify("RUN|DONE\n"), [])


if __name__ == "__main__":
    unittest.main()
