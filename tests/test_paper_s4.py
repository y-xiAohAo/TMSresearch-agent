#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S4 退化检测测试：连续无进展早退强制合成。"""

from __future__ import annotations

import json
import unittest

from research_agent.literature.paper_agent import PaperUnderstandingAgent, _PaperDoc
from research_agent.literature.schemas import PaperAgentConfig

_PAGES = {1: "T\nAbstract.", 2: "3. Methods\nfigure-8 coil radius 0.05 m, 2cm above head. NSGA2 pop 40 gen 60.", 3: "R."}

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

    def __call__(self, messages, tools=None):
        self.calls += 1
        if self.calls > len(self._responses):
            return _resp(_assistant())
        return self._responses[self.calls - 1]


class StallDetectionTests(unittest.TestCase):
    def test_stall_triggers_early_force_synthesize(self):
        # 读一次后，连续 search（无新页无新字段），max_stall=2 → 第 3 轮早退合成
        responses = [
            _resp(_assistant(), [_tc("c1", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),  # 有进展
            _resp(_assistant(), [_tc("c2", "search_text", {"arxiv_id": "x", "query": "zzz"})]),  # 无进展 stall1
            _resp(_assistant(), [_tc("c3", "search_text", {"arxiv_id": "x", "query": "yyy"})]),  # 无进展 stall2 → 触发
            _resp(_assistant(), [_tc("c4", "finish", {"fields": _FIELDS_FULL, "confidence": 2, "sufficient": False})]),  # 合成阶段
        ]
        agent = PaperUnderstandingAgent(
            llm_chat=_ScriptedLLM(responses),
            config=PaperAgentConfig(max_turns=8, max_stall=2, max_searches=10, force_synthesize=True),
        )
        agent._doc = _PaperDoc("x", "v1", _PAGES, [])
        out = agent.run("x")
        self.assertEqual(out["status"], "insufficient")
        # 应在第 3 轮就触发，而非走满 8 轮
        self.assertLessEqual(len(agent._trace), 3)

    def test_progress_resets_stall_count(self):
        responses = [
            _resp(_assistant(), [_tc("c1", "search_text", {"arxiv_id": "x", "query": "zzz"})]),  # 无进展 stall1
            _resp(_assistant(), [_tc("c2", "read_pages", {"arxiv_id": "x", "start": 2, "end": 2})]),  # 有进展 → 重置
            _resp(_assistant(), [_tc("c3", "finish", {"fields": _FIELDS_FULL, "confidence": 4, "sufficient": True})]),
        ]
        agent = PaperUnderstandingAgent(
            llm_chat=_ScriptedLLM(responses),
            config=PaperAgentConfig(max_turns=8, max_stall=2, max_searches=10, force_synthesize=True),
        )
        agent._doc = _PaperDoc("x", "v1", _PAGES, [])
        out = agent.run("x")
        self.assertEqual(out["status"], "ok")


if __name__ == "__main__":
    unittest.main()
