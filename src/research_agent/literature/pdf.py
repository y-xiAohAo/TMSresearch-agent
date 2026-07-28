#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""pdf.py：pymupdf 按页文本提取（供 agent 读论文指定页）。

非语义入库——只按页读取返回文本，全文不进向量库不进 git。
"""

from __future__ import annotations

from pathlib import Path


def read_pdf_pages(pdf_path: str, page_range: str = "1-3", max_chars: int = 8000) -> dict:
    """读取 PDF 指定页范围的文本。

    page_range 形如 "1-3" 或 "2" 或 "1,3,5"。返回 {text, pages_read}。
    """
    import fitz  # pymupdf

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")

    pages = _parse_page_range(page_range)
    text_parts: list[str] = []
    pages_read: list[int] = []
    with fitz.open(str(path)) as doc:
        total = doc.page_count
        for p in pages:
            if 1 <= p <= total:
                page = doc.load_page(p - 1)
                text_parts.append(f"--- Page {p} ---\n{page.get_text('text')}")
                pages_read.append(p)
    text = "\n".join(text_parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return {"text": text, "pages_read": pages_read}


def _parse_page_range(page_range: str) -> list[int]:
    """解析 "1-3" / "2" / "1,3,5" 为页码列表（1-based）。"""
    pages: list[int] = []
    for part in str(page_range).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                start, end = int(start_s), int(end_s)
                pages.extend(range(max(1, start), end + 1))
            except ValueError:
                continue
        else:
            try:
                pages.append(int(part))
            except ValueError:
                continue
    return pages or [1]


def pdf_page_count(pdf_path: str) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count
