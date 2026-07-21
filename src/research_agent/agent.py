#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research Agent：基于 litmusAgent 引擎的科研 ReAct 循环装配。

复用 litmusAgent 的 Agent 引擎（ReAct 循环、trace、错误分类）与 OpenAIClient，
注册本项目的 6 个科研工具（tools.enabled=[] 关闭其默认沙箱工具集）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from research_agent.config import SETTINGS
from research_agent.prompts import RESEARCH_SYSTEM_PROMPT
from research_agent.tools import check_requirements, register_all_tools
from research_agent.tracking import track_tool

MAX_TURNS = 14


def _json_result(fn):
    """把工具 handler 的 dict 返回统一序列化为 JSON 字符串（引擎对返回值做字符串化）。"""

    fn = track_tool(fn)

    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, default=str)
        return result

    wrapper.__name__ = getattr(fn, "__name__", "handler")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


def build_agent():
    """装配 Research Agent，返回 (agent, missing_requirements)。"""
    from agent.config import AgentConfig
    from agent.core.engine import Agent
    from agent.core.types import ToolSpec
    from agent.llm.client import OpenAIClient

    missing = check_requirements(strict=False)

    llm = OpenAIClient(
        api_key=SETTINGS.deepseek_api_key,
        model=SETTINGS.chat_model,
        base_url=SETTINGS.deepseek_base_url,
        max_tokens=4096,
        temperature=0.2,
        timeout=60.0,
        max_retries=1,
    )
    # tools.enabled=[]：不注册 litmusAgent 默认工具（sandbox_exec/file_* 等）
    config = AgentConfig()
    config.tools.enabled = []
    agent = Agent(
        llm_client=llm,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        max_turns=MAX_TURNS,
        config=config,
    )
    for desc in register_all_tools():
        agent.tools.register(
            ToolSpec(
                name=desc.spec.name,
                description=desc.spec.description,
                parameters=desc.spec.parameters,
                handler=_json_result(desc.spec.handler),
            )
        )
    return agent, missing


async def run_research(question: str) -> dict[str, Any]:
    """运行一次科研循环，返回 {answer, missing}。"""
    agent, missing = build_agent()
    unavailable = {
        name: lacks for name, lacks in missing.items()
    }
    prompt = question
    if unavailable:
        prompt += (
            "\n\n[系统提示] 以下工具当前不可用（依赖未就绪），请绕开或说明："
            + json.dumps(unavailable, ensure_ascii=False)
        )
    answer = await agent.run(prompt)
    return {"answer": answer, "missing": missing}


def run_research_sync(question: str) -> dict[str, Any]:
    return asyncio.run(run_research(question))
