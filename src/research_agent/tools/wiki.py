#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wiki 记忆层：结构化研究记录的写入与混合语义检索。

每条记录是一个带 frontmatter 的 markdown 文件，存放在项目 wiki/ 目录，
git 管理、人可读、agent 可检索。M2a 起写入时同步 upsert 进本地 chroma
向量索引（`chroma_wiki/`，gitignore），检索为向量语义 + 关键词计数的
RRF 混合排序；chromadb 不可用/无索引时自动回退纯关键词检索。
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from research_agent.descriptor import ToolDescriptor, ToolSpec

logger = logging.getLogger(__name__)

_WIKI_DIR = Path(__file__).resolve().parents[3] / "wiki"
_CHROMA_DIR = Path(__file__).resolve().parents[3] / "chroma_wiki"
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9一-鿿]+")

_RRF_K = 60
_COLLECTION_CACHE: dict = {}


def _get_collection(embedding_function=None):
    """懒加载 chroma collection；chromadb 不可用或出错时返回 None。

    embedding_function 可注入（测试 mock，避免下载模型）；默认使用
    chromadb 自带的 DefaultEmbeddingFunction（ONNX MiniLM）。
    """
    try:
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    except Exception:
        return None
    key = str(_CHROMA_DIR)
    if embedding_function is None and key in _COLLECTION_CACHE:
        return _COLLECTION_CACHE[key]
    try:
        ef = embedding_function if embedding_function is not None else DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(path=key)
        collection = client.get_or_create_collection("wiki", embedding_function=ef)
        if embedding_function is None:
            _COLLECTION_CACHE[key] = collection
        return collection
    except Exception as exc:
        logger.warning("chroma 索引不可用，回退关键词检索: %s", exc)
        return None


def _slugify(title: str, max_len: int = 60) -> str:
    slug = _SLUG_PATTERN.sub("_", title.strip()).strip("_")
    return slug[:max_len] or "untitled"


def _wiki_write(title: str, content: str, tags: list[str] | None = None) -> dict:
    """把一条研究记录写入 wiki（并同步向量索引），返回 {path, title}。"""
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
    text = frontmatter + content.strip() + "\n"
    path.write_text(text, encoding="utf-8")
    try:
        collection = _get_collection()
        if collection is not None:
            collection.upsert(
                ids=[path.name],
                documents=[text],
                metadatas=[{"title": title, "tags": ", ".join(tags), "path": str(path)}],
            )
    except Exception as exc:  # 索引失败仅警告，不阻断写入
        logger.warning("wiki 向量索引更新失败（文件已写入）: %s", exc)
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


def _keyword_ranked(terms: list[str]) -> list[str]:
    """关键词计数排序，返回按命中数降序的路径列表。"""
    scored: list[tuple[int, str]] = []
    for path in sorted(_WIKI_DIR.glob("*.md")):
        text_lower = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits = sum(text_lower.count(t) for t in terms)
        if hits > 0:
            scored.append((hits, str(path)))
    scored.sort(key=lambda x: -x[0])
    return [p for _, p in scored]


def _semantic_ranked(query: str, max_results: int) -> list[str]:
    """向量语义检索，返回按相关度排序的路径列表；任何异常返回 []。"""
    try:
        collection = _get_collection()
        if collection is None:
            return []
        res = collection.query(query_texts=[query], n_results=max(1, max_results * 2))
        ids = (res.get("ids") or [[]])[0]
        paths = []
        for doc_id in ids:
            path = _WIKI_DIR / doc_id
            if path.is_file():
                paths.append(str(path))
        return paths
    except Exception as exc:
        logger.warning("wiki 语义检索失败，回退关键词检索: %s", exc)
        return []


def _make_entry(path_str: str, terms: list[str], score_source: str) -> dict:
    entry = _parse_entry(Path(path_str))
    text = entry.pop("text")
    idx = text.lower().find(terms[0]) if terms else -1
    start = max(0, idx - 60) if idx >= 0 else 0
    entry["snippet"] = text[start : start + 160].replace("\n", " ").strip()
    entry["score_source"] = score_source
    return entry


def _wiki_search(query: str, max_results: int = 5, mode: str = "hybrid") -> dict:
    """混合语义检索 wiki 条目，返回 {entries: [...]}，entry 含 score_source。

    mode="hybrid"：向量语义 + 关键词 RRF 融合；mode="keyword"：纯关键词；
    mode="semantic"：纯语义（不可用时回退关键词）。
    """
    if not _WIKI_DIR.is_dir():
        return {"entries": []}
    terms = [t for t in re.split(r"\s+", query.strip().lower()) if t]
    if not terms:
        return {"entries": []}
    limit = max(1, max_results)
    keyword_paths = _keyword_ranked(terms)
    semantic_paths = _semantic_ranked(query, limit) if mode in ("hybrid", "semantic") else []
    if not semantic_paths:
        entries = [_make_entry(p, terms, "keyword") for p in keyword_paths[:limit]]
        return {"entries": entries}
    if mode == "semantic":
        entries = [_make_entry(p, terms, "semantic") for p in semantic_paths[:limit]]
        return {"entries": entries}
    scores: dict[str, float] = {}
    sources: dict[str, str] = {}
    for rank, p in enumerate(keyword_paths):
        scores[p] = scores.get(p, 0.0) + 1.0 / (_RRF_K + rank + 1)
        sources[p] = "keyword"
    for rank, p in enumerate(semantic_paths):
        scores[p] = scores.get(p, 0.0) + 1.0 / (_RRF_K + rank + 1)
        sources[p] = "both" if p in sources else "semantic"
    ordered = sorted(scores, key=lambda p: -scores[p])[:limit]
    entries = [_make_entry(p, terms, sources[p]) for p in ordered]
    return {"entries": entries}


def _rebuild_index(embedding_function=None) -> dict:
    """全量重建向量索引，返回 {"indexed": int, "skipped": int}。"""
    md_files = sorted(_WIKI_DIR.glob("*.md")) if _WIKI_DIR.is_dir() else []
    collection = _get_collection(embedding_function=embedding_function)
    if collection is None:
        logger.warning("chromadb 不可用，索引重建跳过。")
        return {"indexed": 0, "skipped": len(md_files)}
    indexed = skipped = 0
    for path in md_files:
        try:
            entry = _parse_entry(path)
            collection.upsert(
                ids=[path.name],
                documents=[entry["text"]],
                metadatas=[{"title": entry["title"], "tags": "", "path": str(path)}],
            )
            indexed += 1
        except Exception as exc:
            logger.warning("索引条目失败 %s: %s", path.name, exc)
            skipped += 1
    return {"indexed": indexed, "skipped": skipped}


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
        description="在个人 wiki 知识库中检索历史研究记录（向量+关键词混合语义检索）。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词或自然语言问题"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        handler=_wiki_search,
    ),
    category="knowledge",
    cost_hint="free",
)
