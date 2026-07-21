#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wiki 工具测试。"""

from __future__ import annotations

import unittest

from research_agent.tools import wiki


class WikiTests(unittest.TestCase):
    def setUp(self):
        wiki._WIKI_DIR = self._tmp_dir_setup()

    def _tmp_dir_setup(self):
        import tempfile
        from pathlib import Path

        return Path(tempfile.mkdtemp(prefix="wiki_test_"))

    def test_write_creates_markdown_with_frontmatter(self):
        result = wiki._wiki_write("单环线圈实验", "# 结论\nNSGA2 收敛。", tags=["tms", "coil"])
        from pathlib import Path

        path = Path(result["path"])
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("title: 单环线圈实验", text)
        self.assertIn("tags: [tms, coil]", text)
        self.assertIn("NSGA2 收敛。", text)

    def test_search_finds_written_entry(self):
        wiki._wiki_write("流函数方法笔记", "streamfunction 是线圈设计核心。", tags=["method"])
        wiki._wiki_write("无关记录", "今天天气不错。", tags=["misc"])
        result = wiki._wiki_search("streamfunction 线圈")
        titles = [e["title"] for e in result["entries"]]
        self.assertIn("流函数方法笔记", titles)
        self.assertNotIn("无关记录", titles)

    def test_search_empty_wiki_returns_empty(self):
        result = wiki._wiki_search("anything")
        self.assertEqual(result["entries"], [])

    def test_search_returns_snippet(self):
        wiki._wiki_write("检索测试", "前文铺垫。" * 30 + "关键词在此出现。")
        result = wiki._wiki_search("关键词")
        self.assertEqual(len(result["entries"]), 1)
        self.assertIn("关键词", result["entries"][0]["snippet"])


if __name__ == "__main__":
    unittest.main()
