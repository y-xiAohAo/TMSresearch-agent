#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_cache.py：论文全文解析产物缓存（版本键 + 原子写）。

缓存键 = {arxiv_id}v{version}（PDF 修订即换键自然失效）；
写入用临时文件 + rename 原子替换（修正并发半文件）。
与 arxiv_cache（HTTP 层，24h TTL）分工：本模块是解析产物层。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from research_agent.config import SETTINGS

_HEADING_LINE_RE = re.compile(r"^\d+(\.\d+)*\s")


def _cache_dir() -> Path:
    d = Path(SETTINGS.artifacts_dir) / "paper_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(arxiv_id: str, version: str) -> Path:
    safe = re.sub(r"[^\w.\-]", "_", arxiv_id)
    return _cache_dir() / f"{safe}{version}.json"


def load(arxiv_id: str, version: str) -> dict | None:
    """读取缓存；不存在或损坏返回 None。"""
    path = _cache_path(arxiv_id, version)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def save(arxiv_id: str, version: str, data: dict) -> None:
    """原子写缓存（临时文件 + rename）。"""
    path = _cache_path(arxiv_id, version)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def build_outline(page_texts: dict[int, str], max_chars: int = 4000) -> list[dict]:
    """从全文构造 outline：每页前 2 个非空行 + 疑似编号标题行（仅导航线索，非闸门）。"""
    outline: list[dict] = []
    chars = 0
    for page_no in sorted(page_texts):
        lines = [ln.strip() for ln in page_texts[page_no].splitlines() if ln.strip()]
        clues = lines[:2]
        heading_lines = [ln for ln in lines if _HEADING_LINE_RE.match(ln) and len(ln) < 80][:2]
        for ln in heading_lines:
            if ln not in clues:
                clues.append(ln)
        entry = {"page": page_no, "lines": clues[:4]}
        cost = sum(len(ln) for ln in entry["lines"])
        if chars + cost > max_chars:
            break
        outline.append(entry)
        chars += cost
    return outline


def full_text(page_texts: dict[int, str]) -> str:
    """拼接全文（闸 3 校验用）。"""
    return "\n".join(page_texts[p] for p in sorted(page_texts))
