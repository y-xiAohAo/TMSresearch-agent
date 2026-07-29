#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S3 循环控制层测试：自检闸 / 轮 6 工具收缩 / 强制合成。"""

from __future__ import annotations

import json
import unittest

from research_agent.literature.paper_agent import PaperUnderstandingAgent, _PaperDoc
from research_agent.literature.schemas import PaperAgentConfig

_PAGES = {
    1: "T\nAbstract.",
    2: "3. Methods\nfigure-8 coil radius 0.05 m, 2cm above head. NSGA2 pop 40 gen 60.",
    3: "4. Results.",
}

_FIELDS_FULL = {
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
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _tc(call_id, name, args):
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


def _resp(message, tool_calls=None):
    return {"message": message, "tool_calls": tool_calls or []}


class _ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_tools = None

    def __call__(self, messages, tools=None):
        self.last_tools = tools
        self.calls += 1
        if self.calls > len(self._responses):
            return _resp(_assistant())
        return self._responses[self.calls - 1]


def _agent(responses, cfg=None):
    cfg = cfg or PaperAgentConfig()
    agent = PaperUnderstandingAgent(llm_chat=_ScriptedLLM(responses), config=cfg)
    agent._doc = _PaperDoc("x", "v1", _PAGES, [])
    return agent


class SelfCheckGateTests(unittest.TestCase):
    def test_self_check_message_at_turn(self):
        llm = _ScriptedLLM([
            _resp(_assistant(), [_tc("c1", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c2", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c3", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c4", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c5", "finish", {"fields": _FIELDS_FULL, "confidence": 4, "sufficient": True})]),
        ])
        agent = PaperUnderstandingAgent(llm_chat=llm, config=PaperAgentConfig(self_check_turn=5, max_turns=8))
        agent._doc = _PaperDoc("x", "v1", _PAGES, [])
        out = agent.run("x")
        self.assertEqual(out["status"], "ok")
        self.assertTrue(agent._self_checked)


class ClosingTurnTests(unittest.TestCase):
    def test_tools_restricted_to_finish_at_closing_turn(self):
        llm = _ScriptedLLM([
            _resp(_assistant(), [_tc("c1", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c2", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c3", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c4", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c5", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c6", "finish", {"fields": _FIELDS_FULL, "confidence": 4, "sufficient": True})]),
        ])
        agent = PaperUnderstandingAgent(llm_chat=llm, config=PaperAgentConfig(closing_turn=6, max_turns=8))
        agent._doc = _PaperDoc("x", "v1", _PAGES, [])
        out = agent.run("x")
        self.assertEqual(out["status"], "ok")
        # 第 6 轮时 LLM 收到的 tools 只剩 finish
        names = [t["function"]["name"] for t in llm.last_tools]
        self.assertEqual(names, ["finish"])


class ForceSynthesizeTests(unittest.TestCase):
    def test_budget_exhaust_triggers_best_effort(self):
        # 前 3 轮只读不 finish，耗尽后强制合成
        llm = _ScriptedLLM([
            _resp(_assistant(), [_tc("c1", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c2", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            _resp(_assistant(), [_tc("c3", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),
            # 强制合成阶段：LLM 提交 finish
            _resp(_assistant(), [_tc("c4", "finish", {"fields": _FIELDS_FULL, "confidence": 2, "sufficient": False})]),
        ])
        agent = PaperUnderstandingAgent(
            llm_chat=llm, config=PaperAgentConfig(max_turns=3, closing_turn=99, force_synthesize=True)
        )
        agent._doc = _PaperDoc("x", "v1", _PAGES, [])
        out = agent.run("x")
        self.assertEqual(out["status"], "insufficient")

    def test_no_collected_stays_incomplete(self):
        llm = _ScriptedLLM([_resp(_assistant())])
        agent = PaperUnderstandingAgent(
            llm_chat=llm, config=PaperAgentConfig(max_turns=1, force_synthesize=True)
        )
        agent._doc = _PaperDoc("x", "v1", _PAGES, [])
        out = agent.run("x")
        self.assertEqual(out["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
