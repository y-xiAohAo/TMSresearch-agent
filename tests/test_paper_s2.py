#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S2 finish 契约层测试：fields/confidence/sufficient 校验。"""

from __future__ import annotations

import unittest

from research_agent.literature.paper_agent import PaperUnderstandingAgent, _PaperDoc
from research_agent.literature.schemas import PaperAgentConfig, get_schema

_PAGES = {
    1: "T\nAbstract.",
    2: "3. Methods\nfigure-8 coil radius 0.05 m, 2cm above head. NSGA2 pop 40 gen 60.",
}
_SCHEMA = get_schema("simulation_params")

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

_FIELDS_ALL_NULL = {
    "coil_geometry": {"type": None, "radius_m": None, "position": None},
    "target_field": {"strength_T": None, "focal_depth_m": None, "region": None},
    "simulation": {"solver": None, "mesh_cells": None, "boundary": None},
    "algorithm": {"name": None, "pop_size": None, "n_gen": None, "objectives": None},
    "evidence_quotes": {},
    "confidence": "low",
}


def _agent() -> PaperUnderstandingAgent:
    agent = PaperUnderstandingAgent(llm_chat=lambda m, t=None: {}, config=PaperAgentConfig())
    agent._doc = _PaperDoc("x", "v1", _PAGES, [])
    agent._current_schema = _SCHEMA
    return agent


class SufficientGateTests(unittest.TestCase):
    def test_sufficient_true_with_values_ok(self):
        agent = _agent()
        verdict, err, result = agent._validate_finish(
            {"fields": _FIELDS_FULL, "confidence": 4, "sufficient": True}, _SCHEMA
        )
        self.assertEqual(verdict, "ok")
        self.assertEqual(result["_meta"]["confidence"], 4)

    def test_sufficient_true_all_null_rejected(self):
        agent = _agent()
        verdict, err, _ = agent._validate_finish(
            {"fields": _FIELDS_ALL_NULL, "sufficient": True}, _SCHEMA
        )
        self.assertEqual(verdict, "reject")
        self.assertIn("sufficient", err)

    def test_sufficient_false_all_null_insufficient(self):
        agent = _agent()
        verdict, err, result = agent._validate_finish(
            {"fields": _FIELDS_ALL_NULL, "sufficient": False}, _SCHEMA
        )
        self.assertEqual(verdict, "insufficient")
        self.assertFalse(result["_meta"]["sufficient"])

    def test_sufficient_false_with_values_still_ok(self):
        agent = _agent()
        verdict, _, _ = agent._validate_finish(
            {"fields": _FIELDS_FULL, "sufficient": False}, _SCHEMA
        )
        self.assertEqual(verdict, "insufficient")


class ConfidenceContractTests(unittest.TestCase):
    def test_confidence_out_of_range_rejected(self):
        agent = _agent()
        verdict, err, _ = agent._validate_finish(
            {"fields": _FIELDS_FULL, "confidence": 7, "sufficient": True}, _SCHEMA
        )
        self.assertEqual(verdict, "reject")
        self.assertIn("confidence", err)

    def test_confidence_mapped_from_envelope(self):
        agent = _agent()
        fields = dict(_FIELDS_FULL)
        fields.pop("confidence")  # 字段级缺失 → envelope 3 映射为 medium
        verdict, err, result = agent._validate_finish(
            {"fields": fields, "confidence": 3, "sufficient": True}, _SCHEMA
        )
        self.assertEqual(verdict, "ok")
        self.assertEqual(result["confidence"], "medium")

    def test_legacy_result_shell_compatible(self):
        agent = _agent()
        verdict, _, result = agent._validate_finish({"result": _FIELDS_FULL}, _SCHEMA)
        self.assertEqual(verdict, "ok")


if __name__ == "__main__":
    unittest.main()
