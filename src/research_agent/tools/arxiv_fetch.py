#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""arxiv_fetch 工具：获取单篇 arXiv 文献元数据，可选下载 PDF 归档。"""

from __future__ import annotations

from research_agent.descriptor import ToolDescriptor, ToolSpec
from research_agent.literature.client import ArxivClient
from research_agent.literature.pdf import pdf_page_count


def _arxiv_fetch(arxiv_id: str, download_pdf: bool = False) -> dict:
    """获取单篇文献详情，可选下载 PDF，返回 {status, paper, pdf_path?, pdf_pages?}。"""
    try:
        client = ArxivClient()
        paper = client.fetch(arxiv_id)
    except RuntimeError as exc:
        return {"status": "network_unavailable", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": f"获取失败：{exc}"}
    if paper is None:
        return {"status": "not_found", "arxiv_id": arxiv_id}
    result = {
        "status": "ok",
        "paper": {
            "arxiv_id": paper.arxiv_id,
            "version": paper.version,
            "title": paper.title,
            "authors": paper.authors,
            "published": paper.published,
            "updated": paper.updated,
            "categories": paper.categories,
            "abstract": paper.abstract,
            "abs_url": paper.abs_url,
            "pdf_url": paper.pdf_url,
        },
    }
    if download_pdf and paper.pdf_url:
        try:
            path = client.download_pdf(paper.arxiv_id)
            result["pdf_path"] = str(path)
            result["pdf_pages"] = pdf_page_count(str(path))
        except Exception as exc:  # noqa: BLE001
            result["pdf_error"] = str(exc)
    return result


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="arxiv_fetch",
        description="按 arxiv_id 获取单篇文献的完整元数据（含摘要），可选下载 PDF 到本地归档。",
        parameters={
            "type": "object",
            "properties": {
                "arxiv_id": {"type": "string", "description": "如 2511.00744（可带版本号 v1）"},
                "download_pdf": {"type": "boolean", "default": False},
            },
            "required": ["arxiv_id"],
        },
        handler=_arxiv_fetch,
    ),
    category="literature",
    cost_hint="free",
    requires=["network_arxiv"],
    produces_artifacts=True,
)
