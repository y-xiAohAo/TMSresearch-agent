#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""extract_template.py：模板驱动的两段式仿真参数抽取。

Stage 1 设备分类：论文 → 设备模板（dipole/patch/tms_figure8/general_params）
Stage 2 模板填充：按模板字段表抽取（带单位 + 引句），校验三道闸。
对齐 ChatCFD 配方：schema 约束生成 + 引句校验 + 失败局部重试。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from research_agent.literature.param_model import FieldValue, ParamValue, SimulationParams

_TEMPLATES_PATH = Path(__file__).resolve().parent / "vocab" / "device_templates.json"


def load_templates() -> dict:
    return json.loads(_TEMPLATES_PATH.read_text(encoding="utf-8"))


_IDENTIFY_PROMPT = """你是仿真任务分类员。读论文片段，判断它属于哪种仿真设备类型。

可选模板：
{template_options}

输出 JSON（不要其他文字）：
{{"template": "<模板名 或 general_params>", "reason": "<一句话>"}}

论文片段：
<external_document>
{text}
</external_document>"""

_FILL_PROMPT = """你是仿真参数抽取员。这是一篇关于 {template} 的论文。

按下面的字段表抽取参数。规则：
- 只抽字段表里的参数，不要新增顶层字段。
- **论文措辞和字段名可能不同，要做语义映射**：
  如 "2 × 9 turns" → turns_per_wing=9；"10 AWG" → wire_diameter（对应线径）；
  "100 mm × 100 mm outset turn" → wing_diameter=0.1（单位转 m）。
- 论文未提及的填 {{"value": null}}，禁止编造。
- 每个非 null 参数必须给出原文引句（quote，可含映射依据）。
- 数值带单位（unit，SI 单位）。

字段表：
{field_table}

输出 JSON：
{{
  "fields": {{
    "<字段名>": {{"value": <数值或null>, "unit": "<单位或null>", "quote": "<原文引句>"}}
  }},
  "extra": {{"<模板外但重要的参数>": {{"value":..., "unit":..., "quote":...}}}},
  "confidence": "high|medium|low"
}}

论文片段：
<external_document>
{text}
</external_document>"""


def _field_table_text(fields: list[dict]) -> str:
    lines = []
    for f in fields:
        req = "（必填）" if f.get("required") else ""
        unit = f"，单位 {f['unit']}" if f.get("unit") else ""
        enum = f"，可选值 {f['enum_options']}" if f.get("enum_options") else ""
        lines.append(f"- {f['name']}: 类型 {f['type']}{unit}{enum}{req}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        raise ValueError("输出中未找到 JSON")
    obj, _ = decoder.raw_decode(text[start:])
    return obj


def _grounded(quote: str, norm_text: str) -> bool:
    """引句滑窗匹配（沿用闸 3 逻辑）。"""
    q = re.sub(r"[\(\[]\s*(?:page|p\.|第)\s*\d+\s*(?:页)?\s*[\)\]]", "", str(quote), flags=re.IGNORECASE)
    for frag in re.split(r"\.{3,}|…", q):
        frag = " ".join(frag.split())
        if not frag:
            continue
        words = frag.split()
        window = min(8, len(words))
        if window == 0:
            continue
        for i in range(0, len(words) - window + 1):
            if " ".join(words[i:i + window]) in norm_text:
                return True
    return False


def _identify_template(text: str, templates: dict, llm_chat) -> str:
    options = "\n".join(f"- {name}: {len(spec['fields'])} 字段" for name, spec in templates.items())
    prompt = _IDENTIFY_PROMPT.format(template_options=options, text=text[:4000])
    try:
        data = _extract_json(llm_chat([{"role": "user", "content": prompt}]))
        chosen = str(data.get("template", "general_params"))
    except Exception:
        chosen = "general_params"
    return chosen if chosen in templates else "general_params"


_PARAM_DENSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m\b|µm|um|GHz|MHz|Hz|kHz|A\b|kA|V\b|mT|T\b|AWG|turns?|radius|diameter|width|height|length|thickness|angle|frequency)",
    re.IGNORECASE,
)
_SECTION_HINT_RE = re.compile(
    r"method|design|geometr|parameter|simulation|setup|experiment|coil|antenna|fabricat|winding",
    re.IGNORECASE,
)


def _select_param_dense_text(text: str, budget: int = 12000) -> str:
    """选参数最密集的文本区（而非硬截前 N 字符）。

    按 2000 字符块扫描，按"数字+单位/参数字"与"方法/几何关键词"打分，
    累加高分块至预算，保持原文顺序拼接。
    """
    if len(text) <= budget:
        return text
    chunk = 2000
    scored: list[tuple[float, int, str]] = []
    for i in range(0, len(text), chunk):
        seg = text[i:i + chunk]
        dense = len(_PARAM_DENSE_RE.findall(seg))
        hints = len(_SECTION_HINT_RE.findall(seg))
        score = dense * 1.0 + hints * 0.5
        scored.append((score, i, seg))
    # 至少保留开头（含标题/摘要线索），再按分补满
    head = scored[0] if scored else (0, 0, text[:chunk])
    rest = sorted(scored[1:], key=lambda x: -x[0])
    chosen = [head]
    total = len(head[2])
    for score, i, seg in rest:
        if total >= budget:
            break
        if score <= 0 and total >= budget * 0.6:
            continue
        chosen.append((score, i, seg))
        total += len(seg)
    chosen.sort(key=lambda x: x[1])
    return "\n".join(seg for _, _, seg in chosen)


def extract_params_template(
    paper_text: str,
    llm_chat,
    templates: dict | None = None,
    max_retries: int = 1,
) -> SimulationParams | dict:
    """两段式模板抽取。成功返回 SimulationParams；失败返回 {status, error}。"""
    templates = templates or load_templates()
    norm_text = " ".join(paper_text.split())

    template_name = _identify_template(paper_text, templates, llm_chat)
    if template_name not in templates:
        return {"status": "error", "error": f"无法识别设备类型（{template_name}）", "template": "general_params"}
    spec = templates[template_name]
    field_table = _field_table_text(spec["fields"])
    fill_text = _select_param_dense_text(paper_text)

    last_err = ""
    for attempt in range(max_retries + 1):
        prompt = _FILL_PROMPT.format(template=template_name, field_table=field_table, text=fill_text)
        raw = llm_chat([{"role": "user", "content": prompt}])
        try:
            data = _extract_json(raw)
            params = _to_params(data, template_name, spec, norm_text)
            return params
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                prompt = (
                    f"上次输出未通过校验（{exc}）。请只输出合法 JSON，"
                    "字段必须在字段表内，引句必须是原文句子。\n\n" + prompt
                )
    return {"status": "extraction_failed", "error": last_err, "template": template_name}


def _to_params(data: dict, template_name: str, spec: dict, norm_text: str) -> SimulationParams:
    fields_in: dict = data.get("fields") or {}
    result_fields: dict[str, FieldValue] = {}

    for fdef in spec["fields"]:
        name = fdef["name"]
        entry = fields_in.get(name)
        if not isinstance(entry, dict):
            if fdef.get("required"):
                raise ValueError(f"必填字段缺失：{name}")
            result_fields[name] = FieldValue(value=None)
            continue
        value = entry.get("value")
        quote = str(entry.get("quote", ""))
        # 引句真实性（闸 3）：非 null 字段引句必须在原文
        if value is not None:
            if not quote:
                raise ValueError(f"字段 {name} 非 null 但缺引句")
            if not _grounded(quote, norm_text):
                raise ValueError(f"字段 {name} 引句非原文子串（疑似编造）")
            pv = ParamValue(value=value, unit=entry.get("unit") or fdef.get("unit"))
            result_fields[name] = FieldValue(value=pv, quote=quote, unit=entry.get("unit"))
        else:
            result_fields[name] = FieldValue(value=None)

    # 模板外字段 → extra
    extra = {}
    for k, v in (data.get("extra") or {}).items():
        if isinstance(v, dict) and v.get("value") is not None and v.get("quote"):
            if _grounded(v["quote"], norm_text):
                extra[k] = v

    confidence = data.get("confidence", "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    return SimulationParams(
        template=template_name,
        fields=result_fields,
        extra=extra,
        confidence=confidence,
    )


def params_to_dict(params: SimulationParams) -> dict:
    return asdict(params)
