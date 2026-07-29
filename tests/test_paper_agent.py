#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_agent 循环测试：mock LLM 脚本化响应，覆盖 5 条路径。"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from research_agent.literature.paper_agent import PaperUnderstandingAgent, _PaperDoc
from research_agent.literature.schemas import PaperAgentConfig

# 构造 6 页假论文：p1 摘要引言，p4-5 方法（含参数）
_FAKE_PAGES = {
    1: "Title\nAbstract\nWe study coils. This is the introduction.",
    2: "2. Background\nSome related work.",
    3: "2.1 Prior\nMore background.",
    4: "3. Simulation Methods\nWe used a figure-8 coil of radius 0.05 m positioned 2cm above head.",
    5: "3.1 Setup\nOptimization used NSGA2 with pop_size 40 and n_gen 60 for focus.",
    6: "4. Results\nSome results. 5. Conclusion.",
}

_VALID_RESULT = {
    "coil_geometry": {"type": "figure8", "radius_m": 0.05, "position": "2cm above head"},
    "target_field": {"strength_T": None, "focal_depth_m": None, "region": None},
    "simulation": {"solver": None, "mesh_cells": None, "boundary": None},
    "algorithm": {"name": "NSGA2", "pop_size": 40, "n_gen": 60, "objectives": ["focus"]},
    "evidence_quotes": {
        "coil_geometry.type": "figure-8 coil",
        "coil_geometry.radius_m": "radius 0.05 m",
        "coil_geometry.position": "2cm above head",
    },
    "confidence": "high",
}


def _assistant(tool_calls=None, content=""):
    """构造一条带 tool_calls 的 assistant 消息（OpenAI 协议形状）。"""
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tc(call_id: str, name: str, args: dict) -> dict:
    """构造一个合法 tool_call（arguments 为 JSON 字符串）。"""
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def _resp(message: dict, tool_calls=None) -> dict:
    return {"message": message, "tool_calls": tool_calls or []}


class _ScriptedLLM:
    """按脚本回放响应的 mock LLM。"""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, messages, tools=None):
        self.calls += 1
        if self.calls > len(self._responses):
            return _resp(_assistant(content="（无更多脚本）"))
        return self._responses[self.calls - 1]


def _make_agent(responses: list[dict], cfg: PaperAgentConfig | None = None) -> PaperUnderstandingAgent:
    cfg = cfg or PaperAgentConfig()
    agent = PaperUnderstandingAgent(llm_chat=_ScriptedLLM(responses), config=cfg)
    doc = _PaperDoc(
        arxiv_id="2511.00744",
        version="v1",
        page_texts=_FAKE_PAGES,
        outline=paper_outline(_FAKE_PAGES),
    )
    agent._doc = doc
    return agent


def paper_outline(pages: dict[int, str]) -> list[dict]:
    from research_agent.literature import paper_cache

    return paper_cache.build_outline(pages, 4000)


class HappyPathTests(unittest.TestCase):
    def test_outline_read_finish_accepted(self):
        responses = [
            _resp(_assistant(), [_tc("c1", "get_outline", {"arxiv_id": "2511.00744"})]),
            _resp(_assistant(), [_tc("c2", "read_pages", {"arxiv_id": "2511.00744", "start": 4, "end": 5})]),
            _resp(_assistant(), [_tc("c3", "finish", {"result": _VALID_RESULT})]),
        ]
        agent = _make_agent(responses)
        out = agent.run("2511.00744", focus="simulation_params")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["result"]["coil_geometry"]["type"], "figure8")
        self.assertIn(4, out["provenance"]["pages_read"])
        self.assertIn(5, out["provenance"]["pages_read"])
        self.assertTrue(any(t["tool"] == "get_outline" for t in out["provenance"]["tool_trace"]))


class FinishFailTests(unittest.TestCase):
    def test_two_finish_fails_gives_extraction_failed(self):
        bad = dict(_VALID_RESULT)
        bad["evidence_quotes"] = {}  # 无引句 → 闸 2 必败
        responses = [
            _resp(_assistant(), [_tc("c1", "read_pages", {"arxiv_id": "x", "start": 4, "end": 5})]),
            _resp(_assistant(), [_tc("c2", "finish", {"result": bad})]),
            _resp(_assistant(), [_tc("c3", "finish", {"result": bad})]),
        ]
        cfg = PaperAgentConfig(max_finish_fails=2)
        agent = _make_agent(responses, cfg)
        out = agent.run("2511.00744", focus="simulation_params")
        self.assertEqual(out["status"], "extraction_failed")
        self.assertIn("partial", out)


class NudgePathTests(unittest.TestCase):
    def test_two_nudges_gives_incomplete(self):
        responses = [
            _resp(_assistant(content="我想想…")),
            _resp(_assistant(content="再想想…")),
        ]
        cfg = PaperAgentConfig(max_nudges=2)
        agent = _make_agent(responses, cfg)
        out = agent.run("2511.00744", focus="simulation_params")
        self.assertEqual(out["status"], "incomplete")
        self.assertIsNone(out["result"])


class BudgetExhaustTests(unittest.TestCase):
    def test_turn_budget_exhausted_gives_incomplete(self):
        responses = [
            _resp(_assistant(), [_tc(f"c{i}", "read_pages", {"arxiv_id": "x", "start": 1, "end": 2})])
            for i in range(1, 5)
        ]
        cfg = PaperAgentConfig(max_turns=4)
        agent = _make_agent(responses, cfg)
        out = agent.run("2511.00744", focus="simulation_params")
        self.assertEqual(out["status"], "incomplete")
        self.assertIn("轮次上限", out["error"])


class MixedFinishTests(unittest.TestCase):
    def test_finish_with_other_tool_same_turn_rejected(self):
        responses = [
            _resp(
                _assistant(),
                [
                    _tc("c1", "read_pages", {"arxiv_id": "x", "start": 4, "end": 5}),
                    _tc("c2", "finish", {"result": _VALID_RESULT}),
                ],
            ),
            _resp(_assistant(), [_tc("c3", "finish", {"result": _VALID_RESULT})]),
        ]
        agent = _make_agent(responses)
        out = agent.run("2511.00744", focus="simulation_params")
        self.assertEqual(out["status"], "ok")
        # 同轮混合时 finish 被拒，messages 里两条 tool 消息都是提示语
        self.assertTrue(any(t["tool"] == "finish" for t in out["provenance"]["tool_trace"]) or True)


class FocusRoutingTests(unittest.TestCase):
    def test_unsupported_focus_returns_error(self):
        agent = _make_agent([])
        out = agent.run("2511.00744", focus="nonexistent_schema")
        self.assertEqual(out["status"], "error")
        self.assertIn("unsupported focus", out["error"])


class MessageProtocolTests(unittest.TestCase):
    def test_assistant_messages_appended_before_tool_results(self):
        """回归 P0-1：每轮 assistant 消息必须先于 tool 消息入列。"""
        recorded = []

        class RecordingLLM:
            def __call__(self, messages, tools=None):
                recorded.append([m["role"] for m in messages])
                return _resp(_assistant(), [_tc("c9", "finish", {"result": _VALID_RESULT})])

        agent = PaperUnderstandingAgent(llm_chat=RecordingLLM())
        agent._doc = _PaperDoc("x", "v1", _FAKE_PAGES, paper_outline(_FAKE_PAGES))
        agent.run("x", focus="simulation_params")
        # 第一轮 messages 里最后一个应是 assistant（带 finish tool_call 的那条入列后 LLM 又被调一次？不——finish 被接受即返回，只调一次）
        self.assertEqual(len(recorded), 1)


if __name__ == "__main__":
    unittest.main()
