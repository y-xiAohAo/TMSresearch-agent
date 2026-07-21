#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""web_search 工具：Tavily 搜索封装。

Tavily 返回为 LLM 优化的结果（含正文摘要），直接可供 ReAct 循环消费。
"""

from __future__ import annotations

from research_agent.config import SETTINGS
from research_agent.descriptor import ToolDescriptor, ToolSpec


def _web_search(query: str, max_results: int = 5) -> dict:
    """调用 Tavily 搜索，返回 {results: [{title, url, content}]}。"""
    if not SETTINGS.tavily_api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY（见 .env）。")
    from tavily import TavilyClient

    client = TavilyClient(api_key=SETTINGS.tavily_api_key)
    resp = client.search(query=query, max_results=max(1, min(int(max_results), 10)))
    results = [
        {
            "title": str(item.get("title", "")),
            "url": str(item.get("url", "")),
            "content": str(item.get("content", "")),
        }
        for item in resp.get("results", [])
    ]
    return {"results": results}


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="web_search",
        description="联网搜索科研文献与资料（Tavily），返回标题/链接/正文摘要列表。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词（建议英文）"},
                "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        handler=_web_search,
    ),
    category="literature",
    cost_hint="cheap",
    requires=["tavily_api_key"],
)
