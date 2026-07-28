#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lit_extract_params 工具（创新点 1）：论文 → 仿真参数桥。

从论文中抽取仿真参数（coil/target/simulation/algorithm），
LLM 抽取 + schema 校验 + 原文引句，未提及字段为 null 不编造。
产出可直接喂 tms_optimize / s4l_run_script。
"""

from __future__ import annotations

from dataclasses import asdict

from research_agent.config import SETTINGS
from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature.client import ArxivClient
from research_agent.literature.extract import extract_sim_params
from research_agent.literature.pdf import read_pdf_pages


def _llm_chat(messages: list[dict]) -> str:
    """同步调 DeepSeek（OpenAI 兼容）返回文本。"""
    from openai import OpenAI

    client = OpenAI(api_key=SETTINGS.deepseek_api_key, base_url=SETTINGS.deepseek_base_url)
    resp = client.chat.completions.create(
        model=SETTINGS.chat_model,
        messages=messages,
        temperature=0.0,
        max_tokens=1500,
        timeout=60,
    )
    return resp.choices[0].message.content or ""


def _lit_extract_params(arxiv_id: str) -> dict:
    """从指定论文抽取仿真参数，返回 SimParamExtraction JSON 或 extraction_failed。"""
    try:
        client = ArxivClient()
        paper = client.fetch(arxiv_id)
    except RuntimeError as exc:
        return {"status": "network_unavailable", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"获取论文失败：{exc}"}
    if paper is None:
        return {"status": "not_found", "arxiv_id": arxiv_id}

    # 优先读 PDF 前两页（方法/参数集中），失败则退回摘要（记录降级原因）
    text = ""
    pdf_fallback_reason = ""
    try:
        pdf_path = client.download_pdf(arxiv_id)
        text = read_pdf_pages(str(pdf_path), page_range="1-2", max_chars=6000)["text"]
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("PDF 读取失败，退回摘要：%s", exc)
        pdf_fallback_reason = str(exc)
        text = paper.abstract
    if not text.strip():
        return {"status": "error", "error": "无法获取论文文本"}

    try:
        result = extract_sim_params(text, _llm_chat, max_retries=1)
    except Exception as exc:  # noqa: BLE001
        # LLM 基础设施异常（网络/鉴权/超时）与抽取失败区分开
        return {"status": "llm_unavailable", "error": f"LLM 调用失败：{exc}"}
    if isinstance(result, dict):
        return result  # extraction_failed
    payload = asdict(result)
    payload["status"] = "ok"
    payload["arxiv_id"] = arxiv_id
    payload["title"] = paper.title
    if pdf_fallback_reason:
        payload["pdf_fallback"] = pdf_fallback_reason
    return payload


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="lit_extract_params",
        description=(
            "从 arXiv 论文抽取电磁/线圈仿真参数（线圈几何/目标场/仿真设置/优化算法），"
            "输出结构化 JSON（含原文引句与置信度），可直接作为 tms_optimize 的 problem_spec 参考。"
            "论文未提及的字段为 null。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "如 2511.00744"},
            },
            "required": ["arxiv_id"],
        },
        handler=_lit_extract_params,
    ),
    category="literature",
    cost_hint="cheap",
    requires=["network_arxiv", "deepseek_api_key"],
)
