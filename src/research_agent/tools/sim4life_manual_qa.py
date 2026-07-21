#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sim4life_manual_qa 工具：调 Sim4Life RAG 知识库（FastAPI SSE）。

回答 Sim4Life 界面操作与手册知识，为 agent 的建模/仿真决策提供指导。
依赖：RAG 服务已启动（uvicorn api_fastapi:app，见 Sim4Life-RAG-Helper）。
"""

from __future__ import annotations

import json

import requests

from research_agent.config import SETTINGS
from research_agent.descriptor import ToolDescriptor, ToolSpec


def _healthcheck() -> None:
    """RAG 服务未就绪时明确报错，不静默降级。"""
    try:
        resp = requests.get(f"{SETTINGS.rag_base_url}/healthz", timeout=10)
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"RAG 服务不可达（{SETTINGS.rag_base_url}）：{exc}。"
            "请先在 Sim4Life-RAG-Helper 启动 uvicorn api_fastapi:app。"
        ) from exc
    if not body.get("runtime_ready"):
        raise RuntimeError(f"RAG runtime 未就绪：{body.get('runtime_error', '')}")


def _sim4life_manual_qa(question: str, top_k: int = 6) -> dict:
    """调用 RAG 服务，拼接 SSE answer_token 为完整答案，返回 {answer}。"""
    _healthcheck()
    payload = {
        "question": question,
        "top_k": max(1, min(int(top_k), 12)),
        "chat_model": "deepseek-v4-flash",
        "answer_mode": "tutorial_first",
        "auto_reasoner": False,
        "history": [],
    }
    answer_parts: list[str] = []
    with requests.post(
        f"{SETTINGS.rag_base_url}/v1/chat/stream",
        json=payload,
        stream=True,
        timeout=180,
    ) as resp:
        resp.raise_for_status()
        event_name = None
        for raw in resp.iter_lines(decode_unicode=True):
            line = (raw or "").strip()
            if not line:
                continue
            if line.startswith("event: "):
                event_name = line[len("event: "):]
            elif line.startswith("data: ") and event_name:
                try:
                    data = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                if event_name == "answer_token":
                    answer_parts.append(str(data.get("text", "")))
                elif event_name == "error":
                    raise RuntimeError(f"RAG 链路错误：{data.get('message', '')}")
                event_name = None
    return {"answer": "".join(answer_parts)}


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="sim4life_manual_qa",
        description=(
            "查询 Sim4Life 手册知识库（RAG）：界面操作指引、参数含义、教程步骤。"
            "建模或仿真设置前先用它确认操作路径。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "关于 Sim4Life 使用的问题"},
                "top_k": {"type": "integer", "default": 6, "minimum": 1, "maximum": 12},
            },
            "required": ["question"],
        },
        handler=_sim4life_manual_qa,
    ),
    category="knowledge",
    cost_hint="cheap",
    requires=["rag_service"],
)
