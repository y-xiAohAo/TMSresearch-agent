#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""figure8_synthesis：tms_figure8 论文参数 → BackendTask（特化物理映射）。

声明近似：
- 近似1：figure8 翼形半径 ≈ 绕组球面半径（coil_sphere.radius = wing_diameter/2）
- 近似2：同心球模型，头模在绕组球内（head = coil - focal_depth）
"""

from __future__ import annotations

from research_agent.literature.param_model import SimulationParams
from research_agent.literature.synthesis import BackendTask, register_synthesis


def _val(params: SimulationParams, name: str, default=None):
    fv = params.fields.get(name)
    if fv is not None and fv.present:
        return fv.value.value
    return default


@register_synthesis("tms_figure8")
def figure8_synthesis(params: SimulationParams) -> BackendTask:
    """tms_figure8 SimulationParams → BackendTask。"""
    wing_d = _val(params, "wing_diameter")
    focal_depth = _val(params, "focal_depth_m")
    region = _val(params, "region")
    i_peak = _val(params, "I_peak")
    wire_d = _val(params, "wire_diameter")
    pulse = _val(params, "pulse_waveform")
    turns = _val(params, "turns_per_wing")

    approximations = []
    geometry = {"kind": "wing_pair", "params": {}}
    if wing_d is not None:
        geometry = {
            "kind": "coil_sphere",
            "params": {
                "radius": float(wing_d) / 2.0,
                # 供建模类后端（sim4life）使用的显式翼参数（tms_optimize 忽略它们）
                "wing_diameter": float(wing_d),
                "turns_per_wing": int(turns) if turns is not None else None,
            },
        }
        approximations.append("近似1: figure8 翼形半径≈绕组球面半径（coil_sphere.radius=wing_diameter/2）")
    if focal_depth is not None:
        approximations.append("近似2: 同心球模型（head_radius = coil_radius − focal_depth）")

    # 靶区椭圆：region 缺失时用 TMS 默认聚焦靶区（必须有，否则优化器无目标）
    ellipses = []
    if region is not None:
        ellipses.append({
            "center_theta": 0.5, "center_phi": 0.0,
            "a": 0.2, "b": 0.1, "alpha": 0.0,
            "sampling_interval": 0.05,
            "note": f"region from paper: {region}",
        })
    else:
        ellipses.append({
            "center_theta": 0.5, "center_phi": 0.0,
            "a": 0.2, "b": 0.1, "alpha": 0.0,
            "sampling_interval": 0.05,
            "note": "default TMS focal target (region not extracted)",
        })

    constraints = {}
    if i_peak is not None:
        constraints["max_current_A"] = float(i_peak)
    if wire_d is not None:
        constraints["wire_diameter_mm"] = float(wire_d) * 1000.0
    constraints["pulse_rise_time_us"] = 100.0  # 默认（论文未给时）

    return BackendTask(
        geometry_intent=geometry,
        field_target={
            "region_ellipses": ellipses,
            "depth_m": focal_depth,
        },
        constraints=constraints,
        outputs=["focality", "focal_area", "field_strength"],
        meta={
            "source_paper": params.source_paper,
            "approximations": approximations,
            "topology_note": f"figure8, turns_per_wing={turns}" if turns else "figure8",
            "turns_per_wing": turns,
            "pulse_waveform": pulse,
        },
    )
