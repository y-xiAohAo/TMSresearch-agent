#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仿真参数值类型系统（L0）与模板化参数模型（L1/L2）。

对齐 HFSS 字符串表达式与 s4l (value, units.X) 元组两态。
独立于 models.py（文献实体），避免侵入既有 SimParamExtraction 体系。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParamValue:
    """带单位/枚举的参数值（对齐 HFSS 表达式与 s4l (value,unit) 元组）。"""

    value: object                       # float | str | list
    unit: str | None = None             # UCUM/软件单位（m, mm, Hz, MilliMeters, Amperes）
    kind: str = "quantity"              # quantity | tuple | expression
    enum_options: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        if self.unit:
            return f"{self.value} {self.unit}"
        return str(self.value)


@dataclass
class FieldValue:
    """一个模板字段的抽取结果：值 + 引句溯源。"""

    value: ParamValue | None            # None = 论文未提及
    quote: str = ""
    unit: str | None = None

    @property
    def present(self) -> bool:
        return self.value is not None and self.value.value is not None


@dataclass
class SimulationParams:
    """按设备模板抽取的仿真参数（L1 骨架 + L2 模板字段）。"""

    template: str                       # dipole / patch / tms_figure8 / general_params
    fields: dict[str, FieldValue] = field(default_factory=dict)
    extra: dict = field(default_factory=dict)   # 模板外参数（带引句）
    source_paper: str = ""
    confidence: str = "low"

    def filled(self) -> dict[str, object]:
        """非 null 字段的值表（喂仿真工具用）。"""
        return {k: fv.value for k, fv in self.fields.items() if fv.present}

    def missing_required(self, template_fields: list[dict]) -> list[str]:
        required = {f["name"] for f in template_fields if f.get("required")}
        return [k for k in required if not self.fields.get(k, FieldValue(None)).present]
