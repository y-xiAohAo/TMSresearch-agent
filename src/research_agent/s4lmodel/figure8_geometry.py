#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""figure8 几何发射器（拓扑特化层）。

已 headless 验证的 API（探针 2026-07-31）：
- model.CreateSolidTube(base_center, axis_height, major_radius, minor_radius)
  → 圆环体（torus）：轴沿 axis_height，环心 = base_center + axis_height/2
- 实体命名：tube.Name = "..."（探针 2 确认 RENAME_OK 后启用，失败静默降级）

绕组近似（诚实边界，spec §4）：turns_per_wing 匝 → 同心圆环组，
径向间距 = wire_diameter（中心线到中心线），非真实螺旋。
真实螺旋需 CreateSpline 逐点构造，留作后续增强。
"""

from __future__ import annotations

# 单位：S4L 全部 SI（米）。BackendTask 内 wire_diameter_mm 需换算。


def _fmt(x: float) -> str:
    """数值格式化：避免科学计数法进脚本，保留足够精度。"""
    return f"{float(x):.9g}"


def emit_single_loop(radius: float, wire_diameter: float, name: str = "loop") -> str:
    """B2a：单环圆线圈（一个 torus 实体），环面在 XY 平面、环心在原点。"""
    wr = float(wire_diameter) / 2.0
    return f'''
# --- geometry: single loop (name={name}) ---
_t = model.CreateSolidTube(
    model.Vec3(0.0, 0.0, {_fmt(-wr)}),
    model.Vec3(0.0, 0.0, {_fmt(2 * wr)}),
    {_fmt(radius)}, {_fmt(wr)},
)
try:
    _t.Name = "{name}_turn0"
except Exception:
    pass
'''


def emit_wing_pair(
    wing_diameter: float,
    turns_per_wing: int,
    wire_diameter: float,
    wing_separation: float | None = None,
) -> tuple[str, list[str]]:
    """B2b：figure8 双翼同心圆环组。

    两翼相切为默认（separation = wing_diameter），环面在 XY 平面（z=0）。
    返回 (脚本片段, 预期实体名列表)。匝数过多致内环半径退化时 clamp 并记录。
    """
    wing_r = float(wing_diameter) / 2.0
    wr = float(wire_diameter) / 2.0
    turns = max(1, int(turns_per_wing))
    sep = float(wing_separation) if wing_separation else float(wing_diameter)
    cx = sep / 2.0

    lines = [
        "# --- geometry: figure8 wing pair "
        f"(wing_d={_fmt(wing_diameter)}, turns={turns}, wire_d={_fmt(wire_diameter)}) ---",
    ]
    names: list[str] = []
    for wing, sign in (("wing_l", -1.0), ("wing_r", 1.0)):
        for k in range(turns):
            r = wing_r - k * float(wire_diameter)
            if r <= wr:
                lines.append(
                    f"# NOTE: turn {k} of {wing} clamped "
                    f"(radius {_fmt(r)} <= wire_radius {_fmt(wr)})"
                )
                r = wr * 2.0
            ent = f"{wing}_turn{k}"
            names.append(ent)
            lines.append(
                f"_t = model.CreateSolidTube(\n"
                f"    model.Vec3({_fmt(sign * cx)}, 0.0, {_fmt(-wr)}),\n"
                f"    model.Vec3(0.0, 0.0, {_fmt(2 * wr)}),\n"
                f"    {_fmt(r)}, {_fmt(wr)},\n"
                f")\n"
                f"try:\n"
                f"    _t.Name = \"{ent}\"\n"
                f"except Exception:\n"
                f"    pass"
            )
    return "\n".join(lines) + "\n", names
