#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_compiler：BackendTask → Sim4Life 建模脚本（注册进 B1 CompilerRegistry）。

输出契约（spec §3.2）：
{
  "script_body": str,         # 不含引导头（s4l_write_script 自动加）
  "expected": {"entity_count_min", "entity_names", "smash_path"},
  "notes": [...],             # 近似声明
}

mapping 规则（特化，隔离在本模块）：
- geometry_intent.kind == "coil_sphere" 且 params 含 wing_diameter → figure8 双翼；
  无 wing_diameter 时退化为单环（radius）。
- constraints.wire_diameter_mm → 米（S4L SI）。
- 空气域半径 = 2×翼展包络半径（override 可调）。
"""

from __future__ import annotations

import time
from pathlib import Path

from research_agent.config import SETTINGS
from research_agent.literature.synthesis import BackendTask, register_compiler
from research_agent.s4lmodel import emlf_setup, figure8_geometry, setup


def _default_smash_path() -> str:
    out_dir = Path(SETTINGS.artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"figure8_{int(time.time())}.smash")


@register_compiler("sim4life")
def s4l_compiler(task: BackendTask, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    geom = task.geometry_intent or {}
    params = geom.get("params", {}) or {}
    constraints = task.constraints or {}
    notes = list((task.meta or {}).get("approximations", []))

    wing_d = params.get("wing_diameter")
    radius = params.get("radius", 0.05)
    wire_d = float(constraints.get("wire_diameter_mm", 2.0)) / 1000.0
    turns = params.get("turns_per_wing") or 1

    parts = [setup.emit_header()]
    entity_names: list[str] = []

    if wing_d:
        body, names = figure8_geometry.emit_wing_pair(
            wing_diameter=float(wing_d),
            turns_per_wing=int(turns),
            wire_diameter=wire_d,
            wing_separation=overrides.get("wing_separation"),
        )
        parts.append(body)
        entity_names.extend(names)
        envelope_r = float(wing_d)  # 双翼包络半径 ≈ 翼径
        notes.append("绕组近似: turns_per_wing 匝→同心圆环组（径向间距=wire_diameter），非真实螺旋")
    else:
        parts.append(figure8_geometry.emit_single_loop(
            radius=float(radius), wire_diameter=wire_d,
        ))
        entity_names.append("loop_turn0")
        envelope_r = float(radius) * 2.0

    air_r = float(overrides.get("air_radius", envelope_r * 2.0))
    parts.append(setup.emit_air_domain((0.0, 0.0, 0.0), air_r))
    entity_names.append("air")

    # 材料指派（默认开：绕组→Copper，空气域→Air；可用 overrides 关闭/改材料）
    materials: dict[str, str] = {}
    if overrides.get("assign_materials", True):
        coil_mat = overrides.get("coil_material", "Copper")
        air_mat = overrides.get("air_material", "Air")
        materials = {n: coil_mat for n in entity_names if n != "air"}
        materials["air"] = air_mat
        parts.append(setup.emit_assign_material(materials))

    smash_path = overrides.get("smash_path") or _default_smash_path()

    # B3 仿真级扩展（overrides.with_simulation 开启时才追加；默认路径逐字节不变）
    # 电流源必须挂线框（pitfall #17）：为每匝补发 CreateCircle wire，
    # 半径规则与 figure8_geometry 逐字节一致（含 r <= wr 时 clamp 到 2*wr）。
    sim_cfg = overrides.get("with_simulation")
    if sim_cfg is not None:
        rings: list[dict] = []
        pos_names: list[str] = []
        neg_names: list[str] = []
        wr = wire_d / 2.0
        if wing_d:
            sep = float(overrides.get("wing_separation") or wing_d)
            cx = sep / 2.0
            wing_r = float(wing_d) / 2.0
            for wing, sign, bucket in (("wing_l", -1.0, pos_names),
                                       ("wing_r", 1.0, neg_names)):
                for k in range(max(1, int(turns))):
                    r = wing_r - k * wire_d
                    if r <= wr:
                        r = wr * 2.0
                    rings.append({"name": f"{wing}_turn{k}",
                                  "center": (sign * cx, 0.0, 0.0), "radius": r})
                    bucket.append(f"{wing}_turn{k}_wire")
        else:
            rings.append({"name": "loop_turn0",
                          "center": (0.0, 0.0, 0.0), "radius": float(radius)})
            pos_names.append("loop_turn0_wire")

        wires_body, wire_names = emlf_setup.emit_current_source_wires(rings)
        parts.append(wires_body)
        entity_names.extend(wire_names)
        parts.append(emlf_setup.emit_mqs_simulation(
            sim_name="sim_tms",
            positive_wires=pos_names,
            current_A=float(sim_cfg.get("current_A", 1.0)),
            wire_radius_m=wr,
            negative_wires=neg_names or None,
            freq_hz=float(sim_cfg.get("freq_hz", 3000.0)),
        ))
        notes.append("两翼电流方向假设：wing_l 正向、wing_r IsDirectionReverted=True")
        notes.append("空气域 MQS 'nothing to solve'（pitfall #18）：仿真脚本为资产交付，"
                     "数值验证走基准复算路径（tools/s4l_solve）")

    parts.append(setup.emit_save_and_report(smash_path))
    if sim_cfg is not None:
        parts.append(emlf_setup.emit_solve())

    return {
        "script_body": "\n".join(parts),
        "expected": {
            # document.New() 自带 2 个基线实体（Model, Grid，探针实测）
            "entity_count_min": 2 + len(entity_names),
            "entity_names": entity_names,
            "smash_path": smash_path,
            "materials": materials,
        },
        "notes": notes,
    }
