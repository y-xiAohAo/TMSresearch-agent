#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""arXiv API 客户端：限流令牌桶 + 重试 + Atom XML 解析 + ID 归一化 + 磁盘缓存。

官方 API：https://export.arxiv.org/api/query（限流 ≤1 req/3s，已实测可达）。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote_plus

import requests

from research_agent.config import SETTINGS
from research_agent.literature.models import LiteratureItem, PaperDetails

_API_BASE = "https://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"
_MIN_INTERVAL_S = 3.0          # arXiv 官方限流：≤1 req/3s
_CACHE_TTL_S = 24 * 3600
_ID_VERSION_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")


def normalize_arxiv_id(raw: str) -> tuple[str, str]:
    """归一化 arxiv_id：去版本尾，返回 (id, version)。无法解析时原样返回 id、version='v1'。"""
    raw = raw.strip()
    for prefix in ("arxiv:", "arXiv:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
    match = _ID_VERSION_RE.match(raw)
    if match:
        return match.group(1), match.group(2) or "v1"
    return raw, "v1"


class ArxivClient:
    """arXiv 查询客户端（限流 + 缓存 + 重试）。"""

    def __init__(self, cache_dir: Path | None = None, min_interval: float = _MIN_INTERVAL_S):
        self._last_request_ts = 0.0
        self._min_interval = min_interval
        self._cache_dir = cache_dir or (Path(SETTINGS.artifacts_dir) / "arxiv_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- 限流 ----
    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    # ---- 缓存 ----
    def _cache_key(self, params: dict) -> Path:
        norm = dict(params)
        if "categories" in norm and isinstance(norm["categories"], list):
            norm["categories"] = sorted(norm["categories"])
        raw = json.dumps(norm, sort_keys=True, ensure_ascii=False)
        return self._cache_dir / (hashlib.md5(raw.encode("utf-8")).hexdigest() + ".json")

    def _cache_get(self, key: Path):
        if not key.is_file():
            return None
        try:
            payload = json.loads(key.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if time.time() - payload.get("ts", 0) > _CACHE_TTL_S:
            return None
        return payload.get("data")

    def _cache_put(self, key: Path, data) -> None:
        try:
            key.write_text(
                json.dumps({"ts": time.time(), "data": data}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ---- HTTP ----
    def _get(self, params: dict) -> str:
        self._throttle()
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                resp = requests.get(_API_BASE, params=params, timeout=30)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.0)
        raise RuntimeError(f"arXiv API 请求失败：{last_exc}")

    # ---- 解析 ----
    def _parse_entry(self, entry: ET.Element) -> PaperDetails:
        def _text(tag: str, default: str = "") -> str:
            el = entry.find(f"{{{_ATOM_NS}}}{tag}")
            return (el.text or default).strip() if el is not None and el.text else default

        raw_id = _text("id")
        arxiv_id, version = normalize_arxiv_id(raw_id.rsplit("/", 1)[-1])
        authors = [
            (a.find(f"{{{_ATOM_NS}}}name").text or "").strip()
            for a in entry.findall(f"{{{_ATOM_NS}}}author")
            if a.find(f"{{{_ATOM_NS}}}name") is not None
        ]
        categories = [
            c.get("term", "")
            for c in entry.findall(f"{{{_ATOM_NS}}}category")
            if c.get("term")
        ]
        pdf_url = ""
        for link in entry.findall(f"{{{_ATOM_NS}}}link"):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
                break
        primary_cat = entry.find(f"{{{_ARXIV_NS}}}primary_category")
        if primary_cat is not None and primary_cat.get("term") and primary_cat.get("term") not in categories:
            categories.insert(0, primary_cat.get("term"))
        return PaperDetails(
            arxiv_id=arxiv_id,
            version=version,
            title=" ".join(_text("title").split()),
            authors=authors,
            published=_text("published"),
            updated=_text("updated"),
            categories=categories,
            abs_url=raw_id,
            ids={"arxiv": arxiv_id},
            abstract=" ".join(_text("summary").split()),
            pdf_url=pdf_url,
        )

    # ---- 公开方法 ----
    def search(
        self,
        query: str,
        field: str = "all",
        categories: list[str] | None = None,
        sort_by: str = "relevance",
        max_results: int = 5,
    ) -> list[PaperDetails]:
        """检索 arXiv，返回 PaperDetails 列表。"""
        field = field if field in ("ti", "abs", "au", "all") else "all"
        search_query = f"{field}:{quote_plus(query.strip())}"
        if categories:
            cats = "+OR+".join(f"cat:{c}" for c in categories)
            search_query = f"({search_query})+AND+({cats})"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max(1, min(int(max_results), 20)),
            "sortBy": sort_by if sort_by in ("relevance", "lastUpdatedDate", "submittedDate") else "relevance",
        }
        key = self._cache_key(params)
        cached = self._cache_get(key)
        if cached is not None:
            entries_xml = cached
        else:
            entries_xml = self._get(params)
            self._cache_put(key, entries_xml)
        root = ET.fromstring(entries_xml)
        return [self._parse_entry(e) for e in root.findall(f"{{{_ATOM_NS}}}entry")]

    def fetch(self, arxiv_id: str) -> PaperDetails | None:
        """按 ID 获取单篇详情；不存在返回 None。"""
        norm_id, _ = normalize_arxiv_id(arxiv_id)
        params = {"id_list": norm_id, "max_results": 1}
        key = self._cache_key(params)
        cached = self._cache_get(key)
        if cached is not None:
            entries_xml = cached
        else:
            entries_xml = self._get(params)
            self._cache_put(key, entries_xml)
        root = ET.fromstring(entries_xml)
        entries = root.findall(f"{{{_ATOM_NS}}}entry")
        if not entries:
            return None
        details = self._parse_entry(entries[0])
        # arXiv 对不存在的 ID 返回标题为 "Error" 的 entry
        if details.title.lower().startswith("error"):
            return None
        return details

    def download_pdf(self, arxiv_id: str, dest_dir: Path | None = None) -> Path:
        """下载 PDF 到 dest_dir（默认 artifacts/papers/），返回路径。"""
        norm_id, _ = normalize_arxiv_id(arxiv_id)
        dest_dir = dest_dir or (Path(SETTINGS.artifacts_dir) / "papers")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{norm_id}.pdf"
        if dest.is_file():
            return dest
        self._throttle()
        resp = requests.get(f"https://arxiv.org/pdf/{norm_id}", timeout=60)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"arXiv {norm_id} 未返回有效 PDF")
        dest.write_bytes(resp.content)
        return dest
