#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分层球壳头模发射器（拓扑特化层，B4 单元 2.5）。

配方依据：B4 单元 2 探针定案（2026-08-04，37 轮）——见 spec
mydocs/specs/2026-08-04_phase-b4-head-model.md「单元 2 完整配方」。

近似声明（诚实边界）：三层同心球壳是 TMS 文献常用的标准简化头模，
非真实解剖结构；头皮以 Skin 近似（ITIS 低频 σ=2.0e-4 S/m）。
"""

from __future__ import annotations

# 默认三层（半径米, ITIS 库材料名, 体素优先级）——内层优先级高（同心重叠覆盖规则）
DEFAULT_LAYERS = (
    ("brain", 0.080, "Brain (Grey Matter)", 30),
    ("skull", 0.087, "Skull Cortical", 20),
    ("scalp", 0.092, "Skin", 10),
)


def _fmt(x: float) -> str:
    return f"{float(x):.9g}"


def emit_head_shells(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    layers: tuple | None = None,
) -> tuple[str, list[str]]:
    """三层同心球壳几何（CreateSolidSphere，已 headless 验证）。

    layers: ((name, radius_m, db_material, priority), ...)，默认 DEFAULT_LAYERS。
    返回 (脚本片段, 实体名列表)。创建顺序外→内（优先级由 voxeler settings 控制，
    见 emlf_setup.emit_solve 的 voxeler_layers）。
    """
    layers = layers or DEFAULT_LAYERS
    cx, cy, cz = (float(v) for v in center)
    lines = ["", "# --- geometry: head shells (layered spheres) ---"]
    names: list[str] = []
    for name, r, _mat, _prio in sorted(layers, key=lambda l: -l[1]):  # 大→小创建
        names.append(name)
        lines.append(
            f"_h = model.CreateSolidSphere(\n"
            f"    model.Vec3({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)}), {_fmt(r)},\n"
            f")\n"
            f"try:\n"
            f"    _h.Name = \"{name}\"\n"
            f"except Exception:\n"
            f"    pass"
        )
    return "\n".join(lines) + "\n", names


def head_material_pairs(layers: tuple | None = None) -> list[tuple[str, str]]:
    """(实体名, 库材料名) 对——供 emlf_setup.emit_material_links 使用。"""
    layers = layers or DEFAULT_LAYERS
    return [(name, mat) for name, _r, mat, _prio in layers]


def head_voxeler_layers(layers: tuple | None = None) -> list[tuple[str, int]]:
    """(实体名, 体素优先级) 对——供 emlf_setup.emit_solve 使用。"""
    layers = layers or DEFAULT_LAYERS
    return [(name, prio) for name, _r, _m, prio in layers]
