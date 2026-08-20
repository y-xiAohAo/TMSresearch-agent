#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全量重建 wiki 向量索引（chroma_wiki/）。

用法：.venv/Scripts/python scripts/rebuild_wiki_index.py
首次运行会下载 chromadb 默认 ONNX MiniLM 模型（约 80MB），属正常。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.tools import wiki


def main() -> int:
    result = wiki._rebuild_index()
    print(f"wiki 索引重建完成: indexed={result['indexed']} skipped={result['skipped']}")
    return 0 if result["indexed"] > 0 or result["skipped"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
