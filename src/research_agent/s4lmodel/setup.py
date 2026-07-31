#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""共享 setup 组件（拓扑无关）：空气域、SaveAs+实体报告、材料、仿真设置。

所有函数为纯脚本发射器（参数 → s4l_v1 代码字符串），不 import s4l_v1。
材料/仿真设置发射器基于二次探针确认的 API（emlf 低频求解器，TMS 适用）。
"""

from __future__ import annotations


def _fmt(x: float) -> str:
    return f"{float(x):.9g}"


def emit_header() -> str:
    """脚本主体公共头部：document.New() + import。"""
    return """import s4l_v1.document as document
import s4l_v1.model as model

document.New()
"""


def emit_air_domain(center: tuple[float, float, float], radius: float,
                    name: str = "air") -> str:
    """空气域球（背景介质）。CreateSolidSphere 已 headless 验证。"""
    cx, cy, cz = center
    return f'''
# --- setup: air domain ---
_air = model.CreateSolidSphere(
    model.Vec3({_fmt(cx)}, {_fmt(cy)}, {_fmt(cz)}), {_fmt(radius)},
)
try:
    _air.Name = "{name}"
except Exception:
    pass
'''


def emit_assign_material(mapping: dict[str, str]) -> str:
    """按实体名指派材料（实体.MaterialName 可写，二次探针确认）。

    mapping: {实体名: 材料名}，如 {"wing_l_turn0": "Copper", "air": "Air"}。
    逐个 try/except，赋值结果打印 REPORT|MATERIAL| 供编排层核验。
    """
    lines = ["", "# --- setup: material assignment ---",
             "_mat_map = {"]
    for ent, mat in mapping.items():
        lines.append(f'    "{ent}": "{mat}",')
    lines += [
        "}",
        "for _e in model.AllEntities():",
        "    _n = getattr(_e, 'Name', '')",
        "    if _n in _mat_map:",
        "        try:",
        "            _e.MaterialName = _mat_map[_n]",
        "            print('REPORT|MATERIAL|' + _n + '|' + str(_e.MaterialName))",
        "        except Exception as _exc:",
        "            print('REPORT|MATERIAL|' + _n + '|FAIL:' + str(_exc))",
    ]
    return "\n".join(lines) + "\n"


def emit_save_and_report(smash_path: str) -> str:
    """SaveAs + 实体报告打印（REPORT| 前缀供编排层断言）。"""
    return f'''
# --- setup: save + entity report ---
document.SaveAs(r"{smash_path}")
_ents = model.AllEntities()
print("REPORT|ENTITY_COUNT|" + str(len(_ents)))
for _e in _ents:
    print("REPORT|ENTITY|" + str(getattr(_e, "Name", "?")))
print("REPORT|DONE")
'''
