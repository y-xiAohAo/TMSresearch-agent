#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用 synthesis 底盘：BackendTask 中间表示 + 两个注册表。

通用部分（与设备/后端无关）：
- BackendTask：参数化的"要做什么仿真"的通用描述。
- SynthesisRegistry：template_name -> synthesis_fn（设备特化在此）。
- CompilerRegistry：backend_name -> compiler_fn（后端特化在此）。

特化隔离在各 synthesis_fn / compiler_fn 内，底盘零改动即可扩展。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class BackendTask:
    """通用中间表示：一次仿真任务的参数化描述（与具体设备/后端无关）。"""

    geometry_intent: dict = field(default_factory=dict)
    # {kind: "coil_sphere"|"wing_pair"|"patch"|"dipole"|..., params: {...}}
    field_target: dict = field(default_factory=dict)
    # {region_ellipses: [...], strength?, depth_m?}
    constraints: dict = field(default_factory=dict)
    # {max_current_A?, wire_diameter_mm?, pulse_rise_time_us?, ...}
    outputs: list[str] = field(default_factory=list)
    # 期望输出指标，如 ["focality", "field_strength", "s11"]
    meta: dict = field(default_factory=dict)
    # {source_paper?, approximations: [...], topology_note?}


# ---------------- SynthesisRegistry ----------------

_SYNTHESES: dict[str, Callable[[Any], BackendTask]] = {}


def register_synthesis(name: str):
    """注册 template_name -> synthesis_fn 的装饰器。"""
    def deco(fn):
        _SYNTHESES[name] = fn
        return fn
    return deco


def get_synthesis(name: str):
    return _SYNTHESES.get(name)


def list_syntheses() -> list[str]:
    return sorted(_SYNTHESES.keys())


# ---------------- CompilerRegistry ----------------

_COMPILERS: dict[str, Callable[[BackendTask, dict], dict]] = {}


def register_compiler(name: str):
    """注册 backend_name -> compiler_fn 的装饰器。"""
    def deco(fn):
        _COMPILERS[name] = fn
        return fn
    return deco


def get_compiler(name: str):
    return _COMPILERS.get(name)


def list_compilers() -> list[str]:
    return sorted(_COMPILERS.keys())


# 触发内置 synthesis/compiler 的装饰器注册（import 副作用）
from research_agent.literature.synthesis import tms_figure8 as _tms_figure8  # noqa: F401,E402
from research_agent.literature.synthesis import tms_optimize_compiler as _tms_compiler  # noqa: F401,E402
