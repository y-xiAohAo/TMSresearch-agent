#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模板抽取测试：两段式（identify + fill + 三道闸）。"""

from __future__ import annotations

import json
import unittest

from research_agent.literature import extract_template
from research_agent.literature.param_model import SimulationParams

_PATCH_PAPER = (
    "A microstrip patch antenna is designed. The substrate is FR4 epoxy with "
    "substrate_height of 1.575 mm and permittivity 4.4. The patch_length is 9.57 mm "
    "and patch_width 9.25 mm, resonating at frequency of 10 GHz."
)

_TEMPLATES = {
    "patch": {
        "fields": [
            {"name": "substrate_height", "type": "quantity", "unit": "m", "required": True},
            {"name": "substrate_permittivity", "type": "float"},
            {"name": "patch_length", "type": "quantity", "unit": "m", "required": True},
            {"name": "patch_width", "type": "quantity", "unit": "m"},
            {"name": "frequency", "type": "quantity", "unit": "Hz", "required": True},
        ],
        "source": "test",
    },
    "dipole": {
        "fields": [{"name": "dipole_length", "type": "quantity", "unit": "m", "required": True}],
        "source": "test",
    },
}


class IdentifyTests(unittest.TestCase):
    def test_identify_patch(self):
        chosen = extract_template._identify_template(
            _PATCH_PAPER, _TEMPLATES, lambda m: '{"template": "patch", "reason": "microstrip"}'
        )
        self.assertEqual(chosen, "patch")

    def test_identify_fallback_general(self):
        chosen = extract_template._identify_template(
            _PATCH_PAPER, _TEMPLATES, lambda m: '{"template": "nonexistent"}'
        )
        self.assertEqual(chosen, "general_params")

    def test_identify_bad_json_fallback(self):
        chosen = extract_template._identify_template(
            _PATCH_PAPER, _TEMPLATES, lambda m: "not json"
        )
        self.assertEqual(chosen, "general_params")


def _fill_llm(data: dict, template: str = "patch"):
    """路由 mock：identify 提示返回分类，fill 提示返回抽取结果。"""
    def llm(messages):
        prompt = messages[0]["content"] if messages else ""
        if "仿真任务分类员" in prompt:
            return json.dumps({"template": template, "reason": "mock"})
        return json.dumps(data)
    return llm


class FillTests(unittest.TestCase):
    _GOOD = {
        "fields": {
            "substrate_height": {"value": 0.001575, "unit": "m", "quote": "substrate_height of 1.575 mm"},
            "substrate_permittivity": {"value": 4.4, "unit": None, "quote": "permittivity 4.4"},
            "patch_length": {"value": 0.00957, "unit": "m", "quote": "patch_length is 9.57 mm"},
            "patch_width": {"value": 0.00925, "unit": "m", "quote": "patch_width 9.25 mm"},
            "frequency": {"value": 10000000000, "unit": "Hz", "quote": "frequency of 10 GHz"},
        },
        "extra": {"substrate": {"value": "FR4 epoxy", "quote": "The substrate is FR4 epoxy"}},
        "confidence": "high",
    }

    def test_valid_extraction(self):
        result = extract_template.extract_params_template(
            _PATCH_PAPER, _fill_llm(self._GOOD), templates=_TEMPLATES
        )
        self.assertIsInstance(result, SimulationParams)
        self.assertEqual(result.template, "patch")
        self.assertTrue(result.fields["substrate_height"].present)
        self.assertEqual(result.fields["substrate_height"].value.value, 0.001575)
        self.assertEqual(result.confidence, "high")
        self.assertIn("substrate", result.extra)

    def test_missing_required_field_rejected(self):
        bad = dict(self._GOOD)
        bad["fields"] = {k: v for k, v in self._GOOD["fields"].items() if k != "substrate_height"}
        result = extract_template.extract_params_template(
            _PATCH_PAPER, _fill_llm(bad), templates=_TEMPLATES, max_retries=0
        )
        self.assertEqual(result["status"], "extraction_failed")
        self.assertIn("substrate_height", result["error"])

    def test_fabricated_quote_rejected(self):
        bad = json.loads(json.dumps(self._GOOD))
        bad["fields"]["patch_length"]["quote"] = "a completely invented sentence not in paper"
        result = extract_template.extract_params_template(
            _PATCH_PAPER, _fill_llm(bad), templates=_TEMPLATES, max_retries=0
        )
        self.assertEqual(result["status"], "extraction_failed")
        self.assertIn("patch_length", result["error"])

    def test_missing_quote_rejected(self):
        bad = json.loads(json.dumps(self._GOOD))
        bad["fields"]["patch_width"]["quote"] = ""
        result = extract_template.extract_params_template(
            _PATCH_PAPER, _fill_llm(bad), templates=_TEMPLATES, max_retries=0
        )
        self.assertEqual(result["status"], "extraction_failed")

    def test_retry_on_bad_json(self):
        calls = []

        def llm(messages):
            prompt = messages[0]["content"] if messages else ""
            if "仿真任务分类员" in prompt:
                return json.dumps({"template": "patch", "reason": "mock"})
            calls.append(1)
            return "garbage" if len(calls) == 1 else json.dumps(self._GOOD)

        result = extract_template.extract_params_template(
            _PATCH_PAPER, llm, templates=_TEMPLATES, max_retries=1
        )
        self.assertIsInstance(result, SimulationParams)
        self.assertEqual(len(calls), 2, "坏 JSON 应重试一次")

    def test_missing_required_list(self):
        result = extract_template.extract_params_template(
            _PATCH_PAPER, _fill_llm(self._GOOD), templates=_TEMPLATES
        )
        self.assertEqual(result.missing_required(_TEMPLATES["patch"]["fields"]), [])


if __name__ == "__main__":
    unittest.main()
