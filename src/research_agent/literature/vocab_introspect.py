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


# s4l_v1.model 的 14 个 Create 原语（从 XCoreModeling docstring dump，2026-07-30 实测）
_S4L_SIGNATURES = {
    "CreatePoint": "position",
    "CreatePolyLine": "points",
    "CreateSpline": "points",
    "CreateArc": "center, radius, start, end",
    "CreateRectangle": "origin, dx, dy",
    "CreateCircle": "center, normal, radius",
    "CreateSolidCylinder": "point1, point2, radius, parametrized=False",
    "CreateSolidTube": "base_center, axis_height, major_radius, minor_radius, parametrized=False",
    "CreateSolidSphere": "center_point, radius",
    "CreateSolidBlock": "p0, p1, parametrized=False",
    "CreateSolidCone": "bottom_center, top_center, bottom_radius, top_radius, parametrized=False",
    "CreateSolidPyramid": "bottom_center, top_center, bottom_radius, top_radius, number_of_sides, start_angle, parametrized=False",
    "CreateWireBlock": "p0, p1, parametrized=False",
    "CreateGroup": "name='Group'",
}


def introspect_s4l() -> dict:
    """s4l 原语：14 个 Create 签名（docstring dump）+ 脚本提取的补充。"""
    prims: dict[str, dict] = {}
    for name, sig in _S4L_SIGNATURES.items():
        params = []
        for part in [p.strip() for p in sig.split(",")]:
            pname = part.split("=")[0].strip()
            has_default = "=" in part
            ptype = "vec3" if any(k in pname for k in ("center", "origin", "point", "normal", "p0", "p1", "bottom", "top")) else "quantity"
            params.append({
                "name": pname, "type": ptype,
                "required": not has_default,
                **({"default": part.split("=", 1)[1].strip()} if has_default else {}),
            })
        prims[name] = {
            "params": params,
            "source": "s4l_v1.model/XCoreModeling_docstring",
            "value_style": "tuple_units",
        }
    # 脚本提取的补充（CreateWireBlock/CreateVoxels/CreatePoint 等脚本实证参数）
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
            if not name:
                continue
            kws = sorted({kw.arg for kw in node.keywords if kw.arg and kw.arg != "parametrized"})
            if name in prims:
                existing = {p["name"] for p in prims[name]["params"]}
                for k in kws:
                    if k not in existing:
                        prims[name]["params"].append({"name": k, "type": "quantity", "required": False})
            else:
                prims[name] = {
                    "params": [{"name": k, "type": "quantity", "required": False} for k in kws],
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


# Antenna Toolkit 天线的字段→单位/类型推断规则
_LENGTH_FIELDS = {
    "length", "width", "height", "radius", "diameter", "thickness", "depth",
    "size", "span", "aperture", "flare", "arm", "element", "feeder_length",
    "substrate_height", "patch_length", "patch_width", "wing_diameter",
    "loop_radius", "wire_diameter", "strip_width", "slot_length", "slot_width",
}
_FREQ_FIELDS = {"frequency", "center_frequency", "resonant_frequency", "start_frequency", "stop_frequency"}
_INT_HINT = ("number", "turns", "count", "sides", "elements", "segments", "points", "mode", "order")


def _infer_field(name: str, default) -> dict:
    """按字段名与默认值推断单位/类型（对齐 Antenna Toolkit 语义）。"""
    lname = name.lower()
    if lname in ("name", "material", "coordinate_system", "outer_boundary"):
        return {"name": name, "type": "string"}
    if lname == "length_unit":
        return {"name": name, "type": "enum", "enum_options": ["mm", "um", "cm", "meter", "mil", "in"], "default": "mm"}
    if lname == "frequency_unit":
        return {"name": name, "type": "enum", "enum_options": ["Hz", "kHz", "MHz", "GHz"], "default": "GHz"}
    if lname in _FREQ_FIELDS or "frequency" in lname:
        return {"name": name, "type": "quantity", "unit": "Hz"}
    if lname == "origin" or (isinstance(default, list) and len(default) == 3):
        return {"name": name, "type": "vec3", "unit": "m"}
    if lname == "material_properties" or isinstance(default, dict):
        return {"name": name, "type": "dict"}
    if any(h in lname for h in _INT_HINT) or isinstance(default, int):
        return {"name": name, "type": "int"}
    if any(h in lname for h in _LENGTH_FIELDS) or isinstance(default, float):
        return {"name": name, "type": "quantity", "unit": "m"}
    return {"name": name, "type": "quantity"}


def build_antenna_toolkit_templates(antenna_fields: dict) -> dict:
    """把 Antenna Toolkit 各天线的字段集批量转为设备模板。"""
    templates = {}
    for antenna, fields in antenna_fields.items():
        tf = []
        for fname in fields:
            spec = _infer_field(fname, None)
            # 关键必填：frequency 与主几何字段
            if fname == "frequency" or fname in ("dipole_length", "patch_length", "wing_diameter", "substrate_height"):
                spec["required"] = True
            tf.append(spec)
        templates[antenna] = {
            "fields": tf,
            "source": "ansys-antenna-toolkit/_default_input_parameters",
        }
    return templates


def main(out_dir: Path | None = None, antenna_fields: dict | None = None) -> dict:
    out_dir = out_dir or (Path(__file__).resolve().parent / "vocab")
    out_dir.mkdir(parents=True, exist_ok=True)

    prims_pyaedt = introspect_pyaedt()
    prims_s4l = introspect_s4l()
    prims = {**prims_pyaedt, **{f"s4l_{k}": v for k, v in prims_s4l.items()}}

    # 设备模板：手维护的优先（dipole/patch/tms_figure8 为抽取精选），
    # Antenna Toolkit 批量只补缺、不覆盖（toolkit 的 dipole/patch 是阵列变体，不适合抽取）
    templates = build_device_templates()
    if antenna_fields:
        for name, spec in build_antenna_toolkit_templates(antenna_fields).items():
            templates.setdefault(name, spec)

    (out_dir / "primitives.json").write_text(
        json.dumps(prims, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "device_templates.json").write_text(
        json.dumps(templates, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"primitives: {len(prims)} (pyaedt {len(prims_pyaedt)}, s4l {len(prims_s4l)})")
    print(f"device_templates: {len(templates)} -> {list(templates.keys())}")
    print(f"out: {out_dir}")
    return {"primitives": prims, "device_templates": templates}


if __name__ == "__main__":
    main()
