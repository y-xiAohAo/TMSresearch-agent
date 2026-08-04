#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解析解模块：B3 的第一 ground truth（不依赖论文、不依赖抽取）。

单环轴线磁场（Biot-Savart 解析式）：
    B(z) = μ₀ · I · R² / (2 · (R² + z²)^{3/2})
同心圆环组：各环独立求和（线性区叠加原理）。
"""

from __future__ import annotations

MU0 = 4e-7 * 3.141592653589793  # 真空磁导率 H/m


def loop_axial_field(radius: float, current: float, z: float) -> float:
    """单匝圆环轴线上距环心 z 处的磁感应强度（Tesla）。"""
    r2 = float(radius) ** 2
    return MU0 * float(current) * r2 / (2.0 * (r2 + float(z) ** 2) ** 1.5)


def concentric_rings_axial_field(radii: list[float], current: float, z: float) -> float:
    """同心圆环组（同电流）轴向场叠加。"""
    return sum(loop_axial_field(r, current, z) for r in radii)


def wing_radii(wing_diameter: float, turns: int, wire_diameter: float) -> list[float]:
    """与 figure8_geometry 同规则的每匝环半径序列（含 clamp 规则）。"""
    wing_r = float(wing_diameter) / 2.0
    wr = float(wire_diameter) / 2.0
    radii = []
    for k in range(max(1, int(turns))):
        r = wing_r - k * float(wire_diameter)
        radii.append(max(r, wr * 2.0))
    return radii
