#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tms_optimize_compiler：BackendTask → tms_optimize ExperimentConfig（特化编译）。"""

from __future__ import annotations

from research_agent.literature.synthesis import BackendTask, register_compiler


@register_compiler("tms_optimize")
def tms_optimize_compiler(task: BackendTask, overrides: dict) -> dict:
    """BackendTask → tms_optimize problem_spec（ExperimentConfig 字段）。"""
    overrides = overrides or {}
    geom = task.geometry_intent.get("params", {})
    target = task.field_target
    cons = task.constraints

    config = {
        "coil_radius": float(overrides.get("coil_radius", geom.get("radius", 0.090))),
        "head_radius": overrides.get("head_radius"),
        "Nsteps": int(overrides.get("Nsteps", 3)),
        "ellipses": target.get("region_ellipses") or overrides.get("ellipses"),
        "nsga2": {
            "pop_size": int(overrides.get("pop_size", 10)),
            "n_gen": int(overrides.get("n_gen", 3)),
            "seed": int(overrides.get("seed", 42)),
        },
        "manufacturability": {},
    }

    # head_radius 由 depth 推导（同心球：head = coil - depth）
    if config["head_radius"] is None:
        depth = target.get("depth_m")
        if depth is not None:
            head = config["coil_radius"] - float(depth)
            config["head_radius"] = max(0.001, head)  # 下限保护
            if head <= 0:
                config["_warn"] = f"head_radius=coil({config['coil_radius']})-depth({depth})≤0，已 clamp 到 0.001"
        else:
            config["head_radius"] = config["coil_radius"]  # 默认同心

    if cons.get("max_current_A") is not None:
        config["manufacturability"]["max_current_A"] = float(cons["max_current_A"])
    if cons.get("wire_diameter_mm") is not None:
        config["manufacturability"]["wire_diameter_mm"] = float(cons["wire_diameter_mm"])
    if cons.get("pulse_rise_time_us") is not None:
        config["pulse_rise_time_us"] = float(cons["pulse_rise_time_us"])

    return config
