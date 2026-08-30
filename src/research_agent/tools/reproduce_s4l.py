#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""reproduce_s4l 工具：论文 → 抽参 → s4l_solve_run（头模建模+GUI 求解+场验证）→ 报告。

顶层薄编排（spec 2026-08-29 §3，方案 A）：
  arxiv_id 路径：literature 层获取/缓存 PDF → 两段式抽参（extract_template，
                 与 reproduce_tms 同一条调用链）→ 必需字段校验；
  params 路径：  调用方直接给结构化参数，绕过 fetch/extract stage；
  assumptions：  论文未报告的值走显式假设入参，合并进最终参数并在报告
                 declarations 逐条声明（禁静默默认填充，防抽参无限重试）；
  solve：        透传 _s4l_solve_run，L2 的分 stage 错误原样上抛；
  报告 + wiki：  参数表 / declarations / 场指标 / verification / 产物路径 /
                 近似声明，write_wiki=True 时沉淀 wiki（失败仅 warning）。
"""

from __future__ import annotations

import time

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature import extract_template
from research_agent.tools import s4l_solve_run
from research_agent.tools import wiki as wiki_tool

# 必需字段：几何三件套（用户 2026-08-29 拍板：缺失 = 抽取未完成，报错重试，禁止静默填充）
_REQUIRED_HINT = (
    "缺失参数=抽取未完成；请用 paper_analyze/lit_extract_params 带 focus 重试，"
    "或确认论文未报告后以 params/assumptions 显式提供"
)

# wire_diameter 单位 → mm 换算系数（抽参模板声明 SI 单位，实测 LLM 会带 m/mm/cm）
_UNIT_TO_MM = {"m": 1000.0, "mm": 1.0, "cm": 10.0, "um": 0.001, "µm": 0.001}


def _fetch_paper_text(arxiv_id: str) -> str:
    """获取论文全文（复用 literature 层缓存/下载，同 reproduce_tms 的调用链）。"""
    from research_agent.literature import paper_cache
    from research_agent.literature.client import normalize_arxiv_id

    norm_id, version = normalize_arxiv_id(arxiv_id)
    cached = paper_cache.load(norm_id, version)
    if cached is None:
        from research_agent.literature.client import ArxivClient

        pdf_path = ArxivClient().download_pdf(norm_id)
        import fitz

        page_texts = {}
        with fitz.open(str(pdf_path)) as doc:
            for i in range(doc.page_count):
                page_texts[i + 1] = doc.load_page(i).get_text("text")
        paper_cache.save(norm_id, version, {"pages": len(page_texts),
                                            "page_texts": {str(k): v for k, v in page_texts.items()}})
        cached = paper_cache.load(norm_id, version)
    return paper_cache.full_text({int(k): v for k, v in cached["page_texts"].items()})


def _extract_params(text: str):
    """两段式模板抽参（复用 extract_template，LLM 通道同 reproduce_tms）。"""
    from research_agent.tools.paper_analyze import _llm_chat

    def llm(messages):
        return _llm_chat(messages, None)["content"]

    return extract_template.extract_params_template(text, llm)


def _field(params, name: str):
    """取抽参字段 (value, unit)；未抽到返回 (None, None)。"""
    fv = params.fields.get(name)
    if fv is not None and fv.present:
        return fv.value.value, fv.value.unit
    return None, None


def _map_extracted(params) -> dict:
    """抽参 SimulationParams → s4l_solve_run 入参表（仅映射抽到的值）。"""
    mapped: dict = {}
    v, _ = _field(params, "wing_diameter")
    if v is not None:
        mapped["wing_diameter"] = float(v)
    v, _ = _field(params, "radius")
    if v is not None:
        mapped["radius"] = float(v)
    v, _ = _field(params, "turns_per_wing")
    if v is not None:
        mapped["turns_per_wing"] = int(v)
    v, unit = _field(params, "wire_diameter")
    if v is not None:
        factor = _UNIT_TO_MM.get(str(unit or "m").lower())
        if factor is None:
            mapped["wire_diameter_unconvertible"] = f"{v} {unit}"
        else:
            mapped["wire_diameter_mm"] = float(v) * factor
    v, _ = _field(params, "I_peak")
    if v is not None:
        mapped["current_A"] = float(v)
    v, _ = _field(params, "freq_hz")
    if v is not None:
        mapped["freq_hz"] = float(v)
    return mapped


def _check_required(solve_params: dict) -> list[str]:
    """必需字段校验：线圈尺寸（翼径或半径）+ 匝数 + 线径。"""
    missing: list[str] = []
    if solve_params.get("wing_diameter") is None and solve_params.get("radius") is None:
        missing.append("wing_diameter 或 radius（线圈尺寸）")
    if solve_params.get("turns_per_wing") is None:
        missing.append("turns_per_wing（匝数）")
    if solve_params.get("wire_diameter_mm") is None:
        if solve_params.get("wire_diameter_unconvertible"):
            missing.append(f"wire_diameter（抽到 {solve_params['wire_diameter_unconvertible']}，单位无法换算成 mm）")
        else:
            missing.append("wire_diameter（线径）")
    return missing


def _build_report(source: str, template: str | None, partial: dict,
                  declarations: list[str], solve_params: dict,
                  solve: dict, duration_s: float) -> str:
    """复现报告（参照 reproduce_tms：参数表/声明/结果/产物/近似声明）。"""
    lines = [
        "# Sim4Life 头模有损求解复现报告",
        "",
        f"- 来源：{source}",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 端到端耗时 = {duration_s} s",
    ]
    if template:
        lines.append(f"- 抽参模板：{template}")
    if partial:
        lines += ["", "## 抽取参数"]
        for k, v in partial.items():
            lines.append(f"- {k} = {v}")
    lines += ["", "## 显式假设声明（declarations）"]
    if declarations:
        for d in declarations:
            lines.append(f"- {d}")
    else:
        lines.append("- 无（全部参数来自论文/显式 params）")
    lines += ["", "## 最终求解参数（s4l_solve_run 入参）"]
    for k in ("wing_diameter", "radius", "turns_per_wing", "wire_diameter_mm",
              "current_A", "freq_hz", "with_head_model"):
        if k in solve_params:
            lines.append(f"- {k} = {solve_params[k]}")
    lines += [
        "",
        "## 求解结果（s4l_solve_run）",
        f"- status = {solve.get('status')}",
    ]
    for name, st in (solve.get("stages") or {}).items():
        lines.append(f"- stage {name}: ok={st.get('ok')} duration={st.get('duration_s')} s")
    fm = solve.get("field_metrics") or {}
    if fm:
        lines += ["", "## 场指标（field_metrics）"]
        for k in ("peak_depth_m", "peak_E", "single_peak", "monotonic_decay"):
            if k in fm:
                lines.append(f"- {k} = {fm[k]}")
        epi = fm.get("E_per_I_at_depths")
        if epi:
            lines.append(f"- E_per_I_at_depths = {epi}")
    ver = solve.get("verification") or {}
    if ver:
        lines += [
            "",
            "## 验证（verification）",
            f"- verify_mode = {ver.get('verify_mode')}",
            f"- verdict = {ver.get('verdict')}",
        ]
        if ver.get("fallback_reason"):
            lines.append(f"- fallback_reason = {ver['fallback_reason']}")
    lines += [
        "",
        "## 产物",
        f"- smash_path = {solve.get('smash_path')}",
        f"- output_h5 = {solve.get('output_h5')}",
        f"- artifacts_size_mb = {solve.get('artifacts_size_mb')}",
        f"- script_path = {solve.get('script_path')}",
        f"- log_path = {solve.get('log_path')}",
        "",
        "## 近似声明",
        "- 几何为同心圆环组近似（非真实螺旋）；头模为三层球壳（脑/颅骨/头皮），非真实解剖头模。",
    ]
    if ver.get("verify_mode") == "reference":
        lines.append(f"- 定量锚：与参照 Output.h5 逐体素对比，阈值 max_rel_err ≤ {ver.get('tolerance', {}).get('ref_pass_max_rel_err')}。")
    else:
        lines.append("- 无参照基准：本报告仅陈述'场型合理 + 量级在判据区间'，不声称与论文数值吻合。")
    return "\n".join(lines) + "\n"


def _reproduce_s4l(
    arxiv_id: str | None = None,
    params: dict | None = None,
    assumptions: dict | None = None,
    with_head_model: bool = True,
    ref_h5: str | None = None,
    write_wiki: bool = True,
    timeout_s: int = 1800,
) -> dict:
    """论文 → 参数 → Sim4Life 头模建模 → GUI 有损求解 → 场验证 一键复现编排。"""
    started = time.perf_counter()

    # ---- 0) 入口校验：arxiv_id 与 params 二选一 ----
    if (arxiv_id is None) == (params is None):
        return {"status": "error", "stage": "input",
                "error": "arxiv_id 与 params 必须二选一（都缺或都给都非法）"}

    # ---- 1) fetch + extract（仅 arxiv_id 路径；params 路径绕过）----
    template = None
    if arxiv_id is not None:
        t0 = time.perf_counter()
        try:
            text = _fetch_paper_text(arxiv_id)
            extracted = _extract_params(text)
        except Exception as exc:
            return {"status": "error", "stage": "extract", "error": f"抽参失败：{exc}"}
        if not hasattr(extracted, "fields"):
            err = extracted.get("error") if isinstance(extracted, dict) else "未知"
            return {"status": "error", "stage": "extract", "error": f"抽取失败：{err}"}
        template = extracted.template
        solve_params = _map_extracted(extracted)
        extract_duration_s = round(time.perf_counter() - t0, 1)
        source = f"arxiv:{arxiv_id}"
    else:
        solve_params = dict(params)
        extract_duration_s = None
        source = "显式 params"

    partial = dict(solve_params)

    # ---- 2) assumptions 显式合并 + 逐条声明 ----
    declarations: list[str] = []
    for k, v in (assumptions or {}).items():
        if k in solve_params and solve_params[k] is not None:
            declarations.append(f"显式假设覆盖抽取值：{k}={v}（抽取值={solve_params[k]}）")
        else:
            declarations.append(f"论文未报告，采用显式假设：{k}={v}")
        solve_params[k] = v

    # ---- 3) 必需字段校验（缺失 = 抽取未完成，报错引导重试，禁静默填充）----
    missing = _check_required(solve_params)
    if missing:
        return {"status": "error", "stage": "extract", "missing": missing,
                "partial": partial, "declarations": declarations,
                "hint": _REQUIRED_HINT}
    solve_params.pop("wire_diameter_unconvertible", None)

    # ---- 4) solve：L2 编排（分 stage 错误原样上抛）----
    solve = s4l_solve_run._s4l_solve_run(
        wing_diameter=solve_params.get("wing_diameter"),
        radius=solve_params.get("radius"),
        turns_per_wing=solve_params.get("turns_per_wing", 1),
        wire_diameter_mm=solve_params.get("wire_diameter_mm", 2.0),
        with_head_model=with_head_model,
        current_A=solve_params.get("current_A", 1.0),
        freq_hz=solve_params.get("freq_hz", 3000.0),
        ref_h5=ref_h5,
        timeout_s=timeout_s,
    )
    if solve.get("status") == "error":
        return {"status": "error", "stage": "solve",
                "solve_stage": solve.get("stage"),
                "stages": solve.get("stages"),
                "error": solve.get("error"),
                "declarations": declarations,
                "solve": solve}
    duration_s = round(time.perf_counter() - started, 1)

    # ---- 5) 报告 + wiki ----
    report = _build_report(source, template, partial, declarations,
                           solve_params, solve, duration_s)
    warnings: list[str] = []
    if write_wiki:
        try:
            wiki_tool._wiki_write(
                title=f"S4L求解复现_{arxiv_id or 'params'}",
                content=report,
                tags=["s4l", "reproduce", "solve", "head_model"],
            )
        except Exception as exc:  # wiki 失败仅 warning，不阻断
            warnings.append(f"wiki 写入失败：{exc}")

    result = {
        "status": solve.get("status", "ok"),
        "source": source,
        "params_template": template,
        "extract_duration_s": extract_duration_s,
        "declarations": declarations,
        "solve_params": {k: v for k, v in solve_params.items() if v is not None},
        "solve": solve,
        "report": report,
        "duration_s": duration_s,
    }
    if solve.get("status") == "failed_verification":
        result["solve_stage"] = "verify"
    if warnings:
        result["warnings"] = warnings
    return result


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="reproduce_s4l",
        description=(
            "论文 → 参数 → Sim4Life 头模建模 → GUI 有损求解 → 场验证 一键复现编排。"
            "arxiv_id 路径走 literature 两段式抽参；抽参缺必需字段（线圈尺寸/匝数/线径）"
            "会报错要求带 focus 重试或以 params/assumptions 显式提供（不静默默认填充）；"
            "论文未报告的值（如 freq_hz/current_A）走显式 assumptions 入参并在报告逐条声明；"
            "求解与验证细节见 s4l_solve_run（分 stage 错误原样上抛）。"
            "license 单座：调用必须严格串行，禁止与其他 S4L 任务并发。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string",
                             "description": "论文 arxiv id（与 params 二选一）"},
                "params": {"type": "object",
                           "description": "显式结构化参数（wing_diameter/radius/turns_per_wing/"
                                          "wire_diameter_mm/current_A/freq_hz），给了就绕过抽参"},
                "assumptions": {"type": "object",
                                "description": "显式假设值（如 {\"freq_hz\": 3000.0}），"
                                               "合并进最终参数并在报告 declarations 逐条声明"},
                "with_head_model": {"type": "boolean", "default": True,
                                    "description": "三层球壳头模；False 为空气域"},
                "ref_h5": {"type": "string",
                           "description": "可选参照 Output.h5（定量锚），透传 s4l_solve_run"},
                "write_wiki": {"type": "boolean", "default": True,
                               "description": "成功后沉淀复现报告到 wiki（失败仅 warning）"},
                "timeout_s": {"type": "integer", "default": 1800,
                              "description": "GUI 求解超时（秒），透传 s4l_solve_run"},
            },
            "required": [],
        },
        handler=_reproduce_s4l,
    ),
    category="simulation",
    cost_hint="expensive",
    async_capable=True,
    requires=["sim4life_gui", "network_arxiv", "deepseek_api_key"],
    produces_artifacts=True,
)
