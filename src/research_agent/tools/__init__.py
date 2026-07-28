#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""工具注册表：手动注册列表 + requires 启动自检。

M2 将改为目录自动扫描（约定导出 DESCRIPTOR），本文件结构无需变化。
"""

from __future__ import annotations

from pathlib import Path

from research_agent.config import SETTINGS
from research_agent.descriptor import ToolDescriptor
from research_agent.tools import (
    arxiv_fetch,
    arxiv_read_pdf,
    arxiv_search,
    lit_extract_params,
    s4l_script,
    sim4life_manual_qa,
    tms_optimize,
    web_search,
    wiki,
)

# 手动注册列表（M2 换自动发现）
ALL_DESCRIPTORS: list[ToolDescriptor] = [
    web_search.DESCRIPTOR,
    sim4life_manual_qa.DESCRIPTOR,
    s4l_script.WRITE_DESCRIPTOR,
    s4l_script.DESCRIPTOR,
    tms_optimize.DESCRIPTOR,
    wiki.DESCRIPTOR,
    wiki.SEARCH_DESCRIPTOR,
    arxiv_search.DESCRIPTOR,
    arxiv_fetch.DESCRIPTOR,
    arxiv_read_pdf.DESCRIPTOR,
    lit_extract_params.DESCRIPTOR,
]

def _check_rag_service() -> bool:
    try:
        import requests

        resp = requests.get(f"{SETTINGS.rag_base_url}/healthz", timeout=5)
        return bool(resp.json().get("runtime_ready"))
    except Exception:
        return False


def _check_network_arxiv() -> bool:
    try:
        import requests

        resp = requests.get("https://export.arxiv.org/api/query?search_query=all:test&max_results=1", timeout=8)
        return resp.status_code == 200
    except Exception:
        return False


# requires 资源名 -> 就绪检查函数
_REQUIRE_CHECKS = {
    "tavily_api_key": lambda: bool(SETTINGS.tavily_api_key),
    "rag_service": _check_rag_service,
    "sim4life_installed": lambda: Path(SETTINGS.s4l_python).is_file() and Path(SETTINGS.s4l_home).is_dir(),
    "tms_venv": lambda: Path(SETTINGS.tms_python).is_file(),
    "network_arxiv": _check_network_arxiv,
    "deepseek_api_key": lambda: bool(SETTINGS.deepseek_api_key),
}


def check_requirements(strict: bool = False) -> dict[str, list[str]]:
    """对所有已注册工具做 requires 自检。

    返回 {工具名: [缺失资源...]}；全部就绪时返回空 dict。
    strict=True 时存在缺失即抛 RuntimeError。
    """
    missing: dict[str, list[str]] = {}
    for desc in ALL_DESCRIPTORS:
        lacks = [
            res
            for res in desc.requires
            if res in _REQUIRE_CHECKS and not _REQUIRE_CHECKS[res]()
        ]
        if lacks:
            missing[desc.spec.name] = lacks
    if missing and strict:
        raise RuntimeError(f"工具依赖未就绪：{missing}")
    return missing


def register_all_tools() -> list[ToolDescriptor]:
    """返回全部已注册工具描述符。"""
    return list(ALL_DESCRIPTORS)
