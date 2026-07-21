#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wiki 记忆层：结构化研究记录的写入与关键词检索。

每条记录是一个带 frontmatter 的 markdown 文件，存放在项目 wiki/ 目录，
git 管理、人可读、agent 可检索。M2 再加向量检索回流。
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from research_agent.descriptor import ToolDescriptor, ToolSpec

_WIKI_DIR = Path(__file__).resolve().parents[3] / "wiki"
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff]+")


def _slugify(title: str, max_len: int = 60) -> str:
    slug = _SLUG_PATTERN.sub("_", title.strip()).strip("_")
    return slug[:max_len] or "untitled"


def _wiki_write(title: str, content: str, tags: list[str] | None = None) -> dict:
    """把一条研究记录写入 wiki，返回 {path, title}。"""
    _WIKI_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = _WIKI_DIR / f"{ts}_{_slugify(title)}.md"
    tags = tags or []
    frontmatter = "---\n" + "\n".join(
        [
            f"title: {title}",
            f"created: {datetime.now().isoformat(timespec='seconds')}",
            "tags: [" + ", ".join(tags) + "]",
        ]
    ) + "\n---\n\n"
    path.write_text(frontmatter + content.strip() + "\n", encoding="utf-8")
    return {"path": str(path), "title": title}


def _parse_entry(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    title = path.stem
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                    break
    return {"title": title, "path": str(path), "text": text}


def _wiki_search(query: str, max_results: int = 5) -> dict:
    """关键词检索 wiki 条目，按命中数排序，返回 {entries: [...]}。"""
    if not _WIKI_DIR.is_dir():
        return {"entries": []}
    terms = [t for t in re.split(r"\s+", query.strip().lower()) if t]
    if not terms:
        return {"entries": []}
    scored: list[tuple[int, dict]] = []
    for path in sorted(_WIKI_DIR.glob("*.md")):
        entry = _parse_entry(path)
        text_lower = entry["text"].lower()
        hits = sum(text_lower.count(t) for t in terms)
        if hits > 0:
            scored.append((hits, entry))
    scored.sort(key=lambda x: -x[0])
    entries = []
    for _, entry in scored[: max(1, max_results)]:
        text = entry.pop("text")
        first_term = terms[0]
        idx = text.lower().find(first_term)
        start = max(0, idx - 60) if idx >= 0 else 0
        entry["snippet"] = text[start : start + 160].replace("\n", " ").strip()
        entries.append(entry)
    return {"entries": entries}


DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="wiki_write",
        description="把一条结构化研究记录（结论/证据/链接）写入个人 wiki 知识库。",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "记录标题"},
                "content": {"type": "string", "description": "markdown 正文"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签列表，可选",
                },
            },
            "required": ["title", "content"],
        },
        handler=_wiki_write,
    ),
    category="knowledge",
    cost_hint="free",
    produces_artifacts=True,
)

SEARCH_DESCRIPTOR = ToolDescriptor(
    spec=ToolSpec(
        name="wiki_search",
        description="在个人 wiki 知识库中按关键词检索历史研究记录。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词（空格分隔）"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        handler=_wiki_search,
    ),
    category="knowledge",
    cost_hint="free",
)
