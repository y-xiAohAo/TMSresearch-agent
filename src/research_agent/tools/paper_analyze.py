#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_analyze 工具：深度理解一篇 arXiv 论文（agent-as-tool）。

主 agent 一次调用 → 内部 PaperUnderstandingAgent 语义导航定位
方法/实验章节，按 focus 对应 schema 抽取结构化参数。
"""

from __future__ import annotations

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature.paper_agent import PaperUnderstandingAgent


def _llm_chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """调 DeepSeek（OpenAI 兼容 chat completions，支持 tools）。"""
    from openai import OpenAI

    from research_agent.config import SETTINGS

    client = OpenAI(api_key=SETTINGS.deepseek_api_key, base_url=SETTINGS.deepseek_base_url)
    kwargs = {
        "model": SETTINGS.chat_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        "timeout": 60,
    }
    if tools:
        kwargs["tools"] = tools
    resp = client.chat.completions.create(**kwargs)
    msg = resp.choices[0].message
    return {
        "message": {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls},
        "content": msg.content or "",
        "tool_calls": (
            [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ]
        ),
    }


def _paper_analyze(arxiv_id: str, focus: str = "simulation_params") -> dict:
    """深度分析论文，返回结构化抽取结果 + 导航轨迹。"""
    agent = PaperUnderstandingAgent(llm_chat=_llm_chat)
    try:
        return agent.run(arxiv_id, focus=focus)
    except Exception as exc:  # noqa: BLE001
        return {"status": "llm_unavailable", "error": f"子代理运行失败：{exc}"}


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="paper_analyze",
        description=(
            "深度理解一篇 arXiv 论文：内部子代理语义导航全文并定位方法/实验章节，"
            "按 focus 指定的 schema 抽取结构化参数（含原文引句、置信度、来源页码）。"
            "比 lit_extract_params（仅前两页浅抽）更深入但更慢。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "如 2511.00744"},
                "focus": {
                    "type": "string",
                    "default": "simulation_params",
                    "description": "抽取目标 schema（当前支持 simulation_params）",
                },
            },
            "required": ["arxiv_id"],
        },
        handler=_paper_analyze,
    ),
    category="literature",
    cost_hint="expensive",
    async_capable=True,
    requires=["network_arxiv", "deepseek_api_key"],
)
