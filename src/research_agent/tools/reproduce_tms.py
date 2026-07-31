#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""reproduce_tms 工具：论文 → 抽参 → synthesis → compiler → tms_optimize 真跑 → 复现报告。

端到端复现闭环（B1）：extract_template → BackendTask → config → tms_optimize。
"""

from __future__ import annotations

import time
from pathlib import Path

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature import extract_template
from research_agent.literature.synthesis import get_compiler, get_synthesis
from research_agent.tools import tms_optimize as tms_tool


def _build_report(arxiv_id: str, params, task, config, metrics, duration_s: float) -> str:
    lines = [
        f"# TMS 复现报告",
        "",
        f"- 论文：{arxiv_id}",
        f"- 模板：{params.template}（confidence={params.confidence}）",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 抽取参数（带引句）",
    ]
    for name, fv in params.fields.items():
        if fv.present:
            lines.append(f"- {name} = {fv.value.value} {fv.value.unit or ''} | {fv.quote[:60]}")
    lines += [
        "",
        "## 映射近似声明",
    ]
    for a in task.meta.get("approximations", []):
        lines.append(f"- {a}")
    lines += [
        "",
        "## 复现配置（ExperimentConfig 摘要）",
        f"- coil_radius = {config.get('coil_radius')} m",
        f"- head_radius = {config.get('head_radius')} m",
        f"- ellipses = {len(config.get('ellipses') or [])} 个靶区",
        f"- max_current_A = {config.get('manufacturability', {}).get('max_current_A')}",
        f"- wire_diameter_mm = {config.get('manufacturability', {}).get('wire_diameter_mm')}",
        "",
        "## 复现结果（tms_optimize NSGA2）",
        f"- status = {metrics.get('status')}",
        f"- n_pareto_solutions = {metrics.get('metrics', {}).get('n_pareto_solutions')}",
        f"- execution_time = {metrics.get('metrics', {}).get('execution_time_sec')} s",
        f"- 端到端耗时 = {duration_s} s",
    ]
    oa = metrics.get("metrics", {}).get("oa90_per_solution")
    if oa:
        lines.append(f"- oa90_per_solution = {oa}")
    fa = metrics.get("metrics", {}).get("focus_areas_mean")
    if fa:
        lines.append(f"- focus_areas_mean = {fa}")
    lines += [
        "",
        "## 局限",
        "- 本复现为 figure8 翼形 → 球面流函数优化器的物理等效近似，非几何级复现。",
        "- 近似与假设见上方'映射近似声明'。",
    ]
    return "\n".join(lines) + "\n"


def _reproduce_tms(arxiv_id: str, budget_s: int = 120) -> dict:
    """端到端复现：论文 → tms_optimize 真跑 → 报告。"""
    started = time.perf_counter()

    # 1. 抽参（用模板抽取，产出 SimulationParams）
    from research_agent.literature import paper_cache
    from research_agent.literature.client import normalize_arxiv_id
    from research_agent.tools.paper_analyze import _llm_chat

    def llm(messages):
        return _llm_chat(messages, None)["content"]

    try:
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
        text = paper_cache.full_text({int(k): v for k, v in cached["page_texts"].items()})
        params = extract_template.extract_params_template(text, llm)
    except Exception as exc:
        return {"status": "error", "error": f"抽参失败：{exc}", "stage": "extract"}

    if not hasattr(params, "fields"):
        return {"status": "error",
                "error": f"抽取失败：{params.get('error') if isinstance(params, dict) else '未知'}",
                "stage": "extract"}

    # 2. synthesis
    synthesis_fn = get_synthesis("tms_figure8")
    if synthesis_fn is None:
        return {"status": "error", "error": "未注册 tms_figure8 synthesis", "stage": "synthesis"}
    task = synthesis_fn(params)

    # 3. compile
    compiler = get_compiler("tms_optimize")
    if compiler is None:
        return {"status": "error", "error": "未注册 tms_optimize compiler", "stage": "compile"}
    config = compiler(task, {})

    # 4. 真跑
    metrics = tms_tool._tms_optimize(problem_spec=config, budget_s=budget_s)

    # 5. 报告
    duration = round(time.perf_counter() - started, 1)
    report = _build_report(arxiv_id, params, task, config, metrics, duration)

    # 6. wiki 沉淀
    try:
        from research_agent.tools import wiki as wiki_tool

        wiki_tool._wiki_write(
            title=f"TMS复现_{arxiv_id}",
            content=report,
            tags=["tms", "reproduce", "figure8", arxiv_id],
        )
    except Exception:
        pass

    return {
        "status": metrics.get("status", "ok"),
        "params_template": params.template,
        "config": config,
        "metrics": metrics,
        "report": report,
        "approximations": task.meta.get("approximations", []),
    }


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="reproduce_tms",
        description=(
            "端到端复现一篇 tms_figure8 论文：抽取参数→映射→tms_optimize 真优化→"
            "产出论文值 vs 复现值对比报告（含近似声明）。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "tms_figure8 论文 arxiv id"},
                "budget_s": {"type": "integer", "default": 120, "description": "优化超时秒数"},
            },
            "required": ["arxiv_id"],
        },
        handler=_reproduce_tms,
    ),
    category="simulation",
    cost_hint="expensive",
    async_capable=True,
    requires=["tms_venv", "network_arxiv"],
    produces_artifacts=True,
)
