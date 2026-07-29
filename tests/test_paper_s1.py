#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S1 遥测与计数层测试：状态行 / 已读去重 / search 计数。"""

from __future__ import annotations

import unittest

from research_agent.literature.paper_agent import PaperUnderstandingAgent, _PaperDoc
from research_agent.literature.schemas import PaperAgentConfig, get_schema

_FAKE_PAGES = {
    1: "Title\nAbstract\nIntro.",
    2: "2. Background\nPrior work.",
    3: "2.1 More\nBackground.",
    4: "3. Methods\nWe used a figure-8 coil of radius 0.05 m.",
    5: "3.1 Setup\nNSGA2 optimization.",
}


def _agent(cfg: PaperAgentConfig | None = None) -> PaperUnderstandingAgent:
    agent = PaperUnderstandingAgent(llm_chat=lambda m, t=None: {}, config=cfg or PaperAgentConfig())
    agent._doc = _PaperDoc("x", "v1", _FAKE_PAGES, [])
    agent._current_schema = get_schema("simulation_params")
    agent._turn = 3
    return agent


class StatusLineTests(unittest.TestCase):
    def test_status_line_format(self):
        agent = _agent()
        agent._pages_read = {1, 4}
        agent._search_count = 2
        line = agent._status_line()
        self.assertIn("Fields=", line)
        self.assertIn("PagesRead=2", line)
        self.assertIn("Searches=2/", line)
        self.assertIn("TurnsLeft=", line)

    def test_dispatch_appends_status_line(self):
        agent = _agent()
        out = agent._dispatch("get_outline", {"arxiv_id": "x"})
        self.assertIn("[Status]", out)


class ReadDedupTests(unittest.TestCase):
    def test_fully_read_pages_deduplicated(self):
        agent = _agent()
        agent._pages_read = {4, 5}
        result = agent._tool_read_pages("x", 4, 5)
        self.assertTrue(result.get("deduplicated"))
        self.assertIn("已", result["text"])
        # 不重复占位
        self.assertEqual(len(agent._collected), 0)

    def test_partial_new_pages_still_read(self):
        agent = _agent()
        agent._pages_read = {4}
        result = agent._tool_read_pages("x", 4, 5)
        self.assertFalse(result.get("deduplicated", False))
        self.assertIn(5, agent._pages_read)


class SearchCountTests(unittest.TestCase):
    def test_search_within_limit(self):
        agent = _agent(PaperAgentConfig(max_searches=2))
        r1 = agent._tool_search_text("x", "coil")
        self.assertGreater(len(r1["hits"]), 0)
        self.assertEqual(agent._search_count, 1)

    def test_search_over_limit_rejected(self):
        agent = _agent(PaperAgentConfig(max_searches=1))
        agent._tool_search_text("x", "coil")
        r2 = agent._tool_search_text("x", "radius")
        self.assertEqual(r2["hits"], [])
        self.assertIn("上限", r2["note"])
        self.assertEqual(agent._search_count, 1, "超限后计数不再增加")


if __name__ == "__main__":
    unittest.main()
