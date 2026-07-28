#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""arxiv_search 工具：arXiv 精确文献检索（分类/日期/作者字段）。

定位：精确文献检索；泛网页检索走 web_search（Tavily）。
"""

from __future__ import annotations

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature.client import ArxivClient

_ABSTRACT_TRUNC = 300


def _arxiv_search(
    query: str,
    field: str = "all",
    categories: list[str] | None = None,
    sort_by: str = "relevance",
    max_results: int = 5,
) -> dict:
    """检索 arXiv，返回 {papers: [...]}；网络失败返回 {status: network_unavailable}。"""
    try:
        client = ArxivClient()
        papers = client.search(
            query=query,
            field=field,
            categories=categories,
            sort_by=sort_by,
            max_results=max_results,
        )
    except RuntimeError as exc:
        return {"status": "network_unavailable", "error": str(exc), "papers": []}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"检索失败：{exc}", "papers": []}
    return {
        "status": "ok",
        "papers": [
            {
                "arxiv_id": p.arxiv_id,
                "version": p.version,
                "title": p.title,
                "authors": p.authors[:5],
                "published": p.published,
                "categories": p.categories,
                "abstract": p.abstract[:_ABSTRACT_TRUNC],
                "abs_url": p.abs_url,
            }
            for p in papers
        ],
    }


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="arxiv_search",
        description=(
            "arXiv 精确文献检索：支持标题/摘要/作者字段、分类过滤（如 physics.med-ph）、"
            "按相关度/日期排序。泛网页检索请用 web_search。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索词（建议英文）"},
                "field": {"type": "string", "enum": ["ti", "abs", "au", "all"], "default": "all"},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "arXiv 分类过滤，如 ['physics.med-ph']",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "lastUpdatedDate", "submittedDate"],
                    "default": "relevance",
                },
                "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        handler=_arxiv_search,
    ),
    category="literature",
    cost_hint="free",
    requires=["network_arxiv"],
)
