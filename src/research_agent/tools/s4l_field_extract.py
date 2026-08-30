#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""s4l_field_extract 工具：Output.h5 焦点场指标提取 + 特征级判定（L3，2026-08-29）。

纯 h5py/numpy 实现（s4lmodel/field_extract），零 license、零 S4L 依赖：
- h5_path 直给，或 results_dir + find_latest_output_h5 自动定位最新输出；
- extract_focal_profile：峰值锚定轴向剖面 → 峰值/深度/单峰/单调衰减指标；
- verify_field_features：判据可配（criteria），缺省用 B4 存量 h5 标定的
  DEFAULT_CRITERIA；表面下 5mm 排除区不参与精确判定（表面点网格敏感 pitfall）。
"""

from __future__ import annotations

import time
from pathlib import Path

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.s4lmodel import field_extract


def _s4l_field_extract(
    h5_path: str | None = None,
    results_dir: str | None = None,
    current_A: float = 1.0,
    criteria: dict | None = None,
) -> dict:
    """从 Output.h5 提取焦点场指标并做特征级判定。"""
    started = time.perf_counter()

    if not h5_path:
        if not results_dir:
            return {"status": "error", "stage": "locate",
                    "error": "h5_path 与 results_dir 至少给一个"}
        h5_path = field_extract.find_latest_output_h5(results_dir)
        if not h5_path:
            return {"status": "error", "stage": "locate",
                    "error": f"结果目录无 *_Output.h5：{results_dir}"}
    if not Path(h5_path).is_file():
        return {"status": "error", "stage": "locate",
                "error": f"Output.h5 不存在：{h5_path}"}

    try:
        metrics = field_extract.extract_focal_profile(h5_path, current_A=current_A)
    except Exception as exc:
        return {"status": "error", "stage": "extract",
                "error": str(exc), "h5_path": h5_path}

    verification = field_extract.verify_field_features(metrics, criteria)
    return {
        "status": "ok",
        "h5_path": h5_path,
        "metrics": metrics,
        "verification": verification,
        "duration_s": round(time.perf_counter() - started, 2),
    }


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="s4l_field_extract",
        description=(
            "从 Sim4Life Output.h5 提取焦点场指标（峰值 E/I、焦点深度、单峰性、"
            "深度单调衰减）并做特征级判定（pass/fail，判据可配，缺省为 B4 真数据"
            "标定的保守量级区间，表面下 5mm 排除区不参与精确判定）。"
            "纯 h5py 解析，零 license、无需 Sim4Life 进程，可分析任意存量结果。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "h5_path": {"type": "string",
                            "description": "Output.h5 路径，缺省时用 results_dir 定位最新"},
                "results_dir": {"type": "string",
                                "description": "<smash>_Results 目录，取 mtime 最新的 *_Output.h5"},
                "current_A": {"type": "number", "default": 1.0,
                              "description": "激励电流（A），指标归一化为 E/I"},
                "criteria": {"type": "object",
                             "description": "可选判据覆盖（键同 DEFAULT_CRITERIA，按论文传专属判据）"},
            },
            "required": [],
        },
        handler=_s4l_field_extract,
    ),
    category="verify",
    cost_hint="cheap",
    requires=[],
    produces_artifacts=False,
)
