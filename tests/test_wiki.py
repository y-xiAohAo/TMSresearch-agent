#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""wiki 工具测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research_agent.tools import wiki


class _FakeCollection:
    """内存版 chroma collection：query 恒空（等价于无索引，走关键词回退）。"""

    def __init__(self):
        self.docs = {}

    def upsert(self, ids, documents, metadatas):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self.docs[doc_id] = (doc, meta)

    def query(self, query_texts, n_results):
        return {"ids": [[]]}


class WikiTests(unittest.TestCase):
    def setUp(self):
        self._orig_wiki_dir = wiki._WIKI_DIR
        self._orig_chroma_dir = wiki._CHROMA_DIR
        self._orig_get_collection = wiki._get_collection
        wiki._WIKI_DIR = Path(tempfile.mkdtemp(prefix="wiki_test_"))
        wiki._CHROMA_DIR = Path(tempfile.mkdtemp(prefix="chroma_test_"))
        # 默认注入 fake collection：测试零网络、零模型下载
        self._fake = _FakeCollection()
        wiki._get_collection = lambda embedding_function=None: self._fake

    def tearDown(self):
        wiki._WIKI_DIR = self._orig_wiki_dir
        wiki._CHROMA_DIR = self._orig_chroma_dir
        wiki._get_collection = self._orig_get_collection

    def test_write_creates_markdown_with_frontmatter(self):
        result = wiki._wiki_write("单环线圈实验", "# 结论\nNSGA2 收敛。", tags=["tms", "coil"])
        path = Path(result["path"])
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("title: 单环线圈实验", text)
        self.assertIn("tags: [tms, coil]", text)
        self.assertIn("NSGA2 收敛。", text)
        # 写入同步进了向量索引
        self.assertEqual(len(self._fake.docs), 1)

    def test_search_finds_written_entry(self):
        wiki._wiki_write("流函数方法笔记", "streamfunction 是线圈设计核心。", tags=["method"])
        wiki._wiki_write("无关记录", "今天天气不错。", tags=["misc"])
        result = wiki._wiki_search("streamfunction 线圈")
        titles = [e["title"] for e in result["entries"]]
        self.assertIn("流函数方法笔记", titles)
        self.assertNotIn("无关记录", titles)
        self.assertEqual(result["entries"][0]["score_source"], "keyword")

    def test_search_empty_wiki_returns_empty(self):
        result = wiki._wiki_search("anything")
        self.assertEqual(result["entries"], [])

    def test_search_returns_snippet(self):
        wiki._wiki_write("检索测试", "前文铺垫。" * 30 + "关键词在此出现。")
        result = wiki._wiki_search("关键词")
        self.assertEqual(len(result["entries"]), 1)
        self.assertIn("关键词", result["entries"][0]["snippet"])

    def test_semantic_recall_with_toy_embedding(self):
        """真实 chroma + 玩具 embedding：中英文近似词跨语言命中，零下载。"""
        try:
            import chromadb
            from chromadb.api.types import EmbeddingFunction
        except ImportError:
            self.skipTest("chromadb 未安装")

        groups = [
            ("streamfunction", "流函数"),
            ("coil", "线圈"),
            ("design", "设计"),
            ("weather", "天气"),
        ]

        class ToyEF(EmbeddingFunction):
            def __call__(self, input):
                vecs = []
                for doc in input:
                    d = doc.lower()
                    vecs.append([float(sum(d.count(w) for w in g)) for g in groups])
                return vecs

        client = chromadb.PersistentClient(path=str(wiki._CHROMA_DIR))
        collection = client.get_or_create_collection("wiki", embedding_function=ToyEF())
        wiki._get_collection = lambda embedding_function=None: collection

        wiki._wiki_write("流函数线圈设计笔记", "用流函数方法做 TMS 线圈设计。", tags=["tms"])
        wiki._wiki_write("无关记录", "今天天气不错 weather。", tags=["misc"])
        result = wiki._wiki_search("streamfunction coil design")
        titles = [e["title"] for e in result["entries"]]
        self.assertIn("流函数线圈设计笔记", titles)
        self.assertEqual(titles[0], "流函数线圈设计笔记")
        entry = next(e for e in result["entries"] if e["title"] == "流函数线圈设计笔记")
        self.assertEqual(entry["score_source"], "semantic")

    def test_search_falls_back_when_chroma_fails(self):
        """chromadb 任意异常：写入不阻断，检索回退关键词且行为不变。"""

        class BrokenCollection:
            def upsert(self, ids, documents, metadatas):
                raise RuntimeError("boom")

            def query(self, query_texts, n_results):
                raise RuntimeError("boom")

        wiki._get_collection = lambda embedding_function=None: BrokenCollection()
        wiki._wiki_write("降级测试", "关键词命中内容。")  # 不抛异常
        result = wiki._wiki_search("关键词")
        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(result["entries"][0]["title"], "降级测试")
        self.assertEqual(result["entries"][0]["score_source"], "keyword")

    def test_rebuild_index(self):
        wiki._wiki_write("条目一", "内容一")
        wiki._wiki_write("条目二", "内容二")
        self._fake.docs.clear()
        result = wiki._rebuild_index()
        self.assertEqual(result["indexed"], 2)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual(len(self._fake.docs), 2)

    def test_rebuild_index_without_chromadb(self):
        wiki._get_collection = lambda embedding_function=None: None
        wiki._wiki_write("条目一", "内容一")
        result = wiki._rebuild_index()
        self.assertEqual(result["indexed"], 0)
        self.assertEqual(result["skipped"], 1)


if __name__ == "__main__":
    unittest.main()
