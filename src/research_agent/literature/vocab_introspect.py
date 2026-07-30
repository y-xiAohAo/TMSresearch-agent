#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""vocab_introspect.py：软件内省驱动的词汇表导出管线。

从本机 PyAEDT 源码（~/.kimi-work/pkg/pyaedt_src/）与 s4l 脚本中
自动导出几何原语词汇表与设备参数模板，产出 JSON。
重跑本管线即可随软件版本更新词汇表。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_PKG = Path.home() / ".kimi-work" / "pkg"
_S4L = Path.home() / ".kimi-work" / "s4l"

# 类型注解 → 词汇表类型映射
_TYPE_MAP = {
    "float": "quantity", "int": "int", "str": "string", "bool": "bool",
    "list": "vec", "tuple": "vec",
}

_ENUM_DOC_RE = re.compile(r"``\"([^\"]+)\"``")


@dataclass
class ParamSpec:
    name: str
    type: str
    required: bool = False
    default: object = None
    unit: str | None = None
    enum_options: list[str] = field(default_factory=list)


def _ann_to_type(ann: ast.expr | None) -> str:
    """把 AST 类型注解映射为词汇表类型（先判 enum/Plane，再判数值类型）。"""
    if ann is None:
        return "quantity"
    text = ast.unparse(ann)
    if "Plane" in text or "Axis" in text:
        return "enum"
    for key, val in _TYPE_MAP.items():
        if key in text:
            return val
    return "quantity"


def _extract_enums_from_doc(doc: str) -> dict[str, list[str]]:
    """从 docstring 的 Parameters 段提取每个参数的枚举词表。"""
    enums: dict[str, list[str]] = {}
    current_param = None
    for line in doc.splitlines():
        m = re.match(r"\s*(\w+)\s*:", line)
        if m and not line.strip().startswith(("Returns", "Yields", "References", "Examples")):
            current_param = m.group(1)
        opts = _ENUM_DOC_RE.findall(line)
        if opts and current_param:
            # 过滤掉单位/示例值，只保留像枚举的短词
            candidates = [o for o in opts if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _\-]{0,30}", o)]
            if candidates:
                enums.setdefault(current_param, [])
                for c in candidates:
                    if c not in enums[current_param]:
                        enums[current_param].append(c)
    return enums


def introspect_pyaedt() -> dict:
    """解析 PyAEDT 源码，导出 create_* 原语签名表。"""
    prims: dict[str, dict] = {}
    for fname in ("primitives_3d.py", "primitives.py", "polylines.py"):
        path = _PKG / "pyaedt_src" / "ansys" / "aedt" / "core" / "modeler" / "cad" / fname
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("create_"):
                continue
            doc = ast.get_docstring(node) or ""
            enums = _extract_enums_from_doc(doc)
            params: list[ParamSpec] = []
            args = node.args
            defaults = list(args.defaults)
            plain = list(args.args) + list(args.kwonlyargs)
            default_map: dict[str, object] = {}
            for a, d in zip(args.args[-len(defaults):] if defaults else [], defaults):
                default_map[a.arg] = _literal(d)
            for a in plain:
                if a.arg in ("self", "kwargs"):
                    continue
                has_default = a.arg in default_map
                params.append(ParamSpec(
                    name=a.arg,
                    type=_ann_to_type(a.annotation),
                    required=not has_default,
                    default=default_map.get(a.arg),
                    enum_options=enums.get(a.arg, []),
                ))
            prims[node.name] = {
                "params": [_param_to_dict(p) for p in params],
                "source": f"pyaedt/{fname}",
                "doc_first_line": doc.strip().splitlines()[0] if doc.strip() else "",
            }
    return prims


def _literal(node: ast.expr):
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if node else None


def _param_to_dict(p: ParamSpec) -> dict:
    d = {"name": p.name, "type": p.type, "required": p.required}
    if p.default is not None:
        d["default"] = p.default
    if p.unit:
        d["unit"] = p.unit
    if p.enum_options:
        d["enum_options"] = p.enum_options
    return d


def introspect_s4l() -> dict:
    """用 AST 从 s4l 脚本提取 model.Create* 调用的关键字参数（正则处理不了嵌套括号）。"""
    prims: dict[str, dict] = {}
    for script in _S4L.rglob("*.py"):
        try:
            tree = ast.parse(script.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Attribute) and func.attr.startswith("Create"):
                name = func.attr
            if not name or name in prims:
                continue
            kws = sorted({kw.arg for kw in node.keywords if kw.arg})
            params = [
                {"name": k, "type": "quantity", "required": False}
                for k in kws if k != "parametrized"
            ]
            prims[name] = {
                "params": params,
                "source": f"s4l_script/{script.name}",
                "value_style": "tuple_units",
            }
    return prims


def build_device_templates() -> dict:
    """组装设备参数模板（天线/线圈直接套，不从原语拼）。"""
    return {
        "dipole": {
            "fields": [
                {"name": "dipole_length", "type": "quantity", "unit": "m", "required": True},
                {"name": "arm_radius", "type": "quantity", "unit": "m"},
                {"name": "gap", "type": "quantity", "unit": "m"},
                {"name": "frequency", "type": "quantity", "unit": "Hz", "required": True},
            ],
            "source": "hfss.get_component_variables(Dipole_Antenna)",
        },
        "patch": {
            "fields": [
                {"name": "substrate_height", "type": "quantity", "unit": "m", "required": True},
                {"name": "substrate_permittivity", "type": "float"},
                {"name": "patch_length", "type": "quantity", "unit": "m", "required": True},
                {"name": "patch_width", "type": "quantity", "unit": "m"},
                {"name": "feed_position_offset", "type": "quantity"},
                {"name": "frequency", "type": "quantity", "unit": "Hz", "required": True},
            ],
            "source": "ansys-antenna-toolkit/_default_input_parameters",
        },
        "tms_figure8": {
            "fields": [
                {"name": "wing_diameter", "type": "quantity", "unit": "m", "required": True},
                {"name": "turns_per_wing", "type": "int", "required": True},
                {"name": "wire_diameter", "type": "quantity", "unit": "m"},
                {"name": "wing_angle_deg", "type": "float"},
                {"name": "I_peak", "type": "quantity", "unit": "A"},
                {"name": "dI_dt", "type": "quantity", "unit": "A/s"},
                {"name": "pulse_waveform", "type": "enum", "enum_options": ["monophasic", "biphasic"]},
            ],
            "source": "tms_literature/s4l_coil_params",
        },
    }


def main(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (Path(__file__).resolve().parent / "vocab")
    out_dir.mkdir(parents=True, exist_ok=True)

    prims_pyaedt = introspect_pyaedt()
    prims_s4l = introspect_s4l()
    prims = {**prims_pyaedt, **{f"s4l_{k}": v for k, v in prims_s4l.items()}}
    templates = build_device_templates()

    (out_dir / "primitives.json").write_text(
        json.dumps(prims, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "device_templates.json").write_text(
        json.dumps(templates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"primitives: {len(prims)} (pyaedt {len(prims_pyaedt)}, s4l {len(prims_s4l)})")
    print(f"device_templates: {list(templates.keys())}")
    print(f"out: {out_dir}")
    return {"primitives": prims, "device_templates": templates}


if __name__ == "__main__":
    main()
