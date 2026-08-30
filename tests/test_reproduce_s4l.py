#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""reproduce_s4l 编排单测：抽参层 / _s4l_solve_run / _wiki_write 三层全 mock，零外部依赖。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from research_agent.literature.param_model import FieldValue, ParamValue, SimulationParams
from research_agent.tools import reproduce_s4l


def _make_params(fields: dict) -> SimulationParams:
    """fields: {name: (value, unit)}，构造两段式抽参的 SimulationParams。"""
    return SimulationParams(
        template="tms_figure8",
        fields={k: FieldValue(value=ParamValue(value=v, unit=u))
                for k, (v, u) in fields.items()},
        confidence="high",
    )


FULL_FIELDS = {
    "wing_diameter": (0.1, "m"),
    "turns_per_wing": (9, None),
    "wire_diameter": (0.0053, "m"),
}

SOLVE_OK = {
    "status": "ok",
    "stages": {"compile": {"ok": True, "duration_s": 0.1},
               "run": {"ok": True, "duration_s": 420.0}},
    "smash_path": "artifacts/x.smash",
    "output_h5": "artifacts/x_Results/x_Output.h5",
    "field_metrics": {"peak_depth_m": 0.001, "peak_E": 6.0e-2,
                      "single_peak": True, "monotonic_decay": True},
    "verification": {"verify_mode": "feature", "verdict": "pass"},
    "artifacts_size_mb": 275.0,
}


def _patch_layers(test, extracted=FULL_FIELDS, solve_result=None):
    """返回 (mock_fetch, mock_extract, mock_solve, mock_wiki) 四层 patcher 上下文。"""
    solve_result = SOLVE_OK if solve_result is None else solve_result
    return (
        patch.object(reproduce_s4l, "_fetch_paper_text", return_value="paper text"),
        patch.object(reproduce_s4l, "_extract_params",
                     return_value=_make_params(extracted) if extracted is not None
                     else {"status": "error", "error": "boom"}),
        patch.object(reproduce_s4l.s4l_solve_run, "_s4l_solve_run",
                     return_value=solve_result),
        patch.object(reproduce_s4l.wiki_tool, "_wiki_write", return_value={"status": "ok"}),
    )


class InputValidationTests(unittest.TestCase):
    def test_both_missing(self):
        r = reproduce_s4l._reproduce_s4l()
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "input")

    def test_both_given(self):
        r = reproduce_s4l._reproduce_s4l(arxiv_id="2511.00744",
                                         params={"wing_diameter": 0.1})
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "input")


class ArxivPathTests(unittest.TestCase):
    def test_happy_path_order_and_passthrough(self):
        fetch, extract, solve, wiki = _patch_layers(self)
        with fetch as m_fetch, extract as m_extract, solve as m_solve, wiki as m_wiki:
            r = reproduce_s4l._reproduce_s4l(
                arxiv_id="2511.00744", assumptions={"freq_hz": 3000.0},
                ref_h5="ref.h5", timeout_s=900)
        m_fetch.assert_called_once_with("2511.00744")
        m_extract.assert_called_once_with("paper text")
        m_solve.assert_called_once_with(
            wing_diameter=0.1, radius=None, turns_per_wing=9,
            wire_diameter_mm=5.3,            # 0.0053 m 换算成 mm
            with_head_model=True, current_A=1.0, freq_hz=3000.0,
            ref_h5="ref.h5", timeout_s=900)
        m_wiki.assert_called_once()
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["params_template"], "tms_figure8")
        self.assertIsNotNone(r["extract_duration_s"])
        self.assertIn("复现报告", r["report"])

    def test_missing_required_blocks_solve(self):
        fetch, extract, solve, wiki = _patch_layers(
            self, extracted={"wing_diameter": (0.1, "m")})  # 缺匝数+线径
        with fetch, extract, solve as m_solve, wiki as m_wiki:
            r = reproduce_s4l._reproduce_s4l(arxiv_id="2511.00744")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "extract")
        self.assertTrue(any("turns_per_wing" in m for m in r["missing"]))
        self.assertTrue(any("wire_diameter" in m for m in r["missing"]))
        self.assertEqual(r["partial"], {"wing_diameter": 0.1})
        self.assertIn("hint", r)
        m_solve.assert_not_called()
        m_wiki.assert_not_called()

    def test_wire_unit_conversion_cm(self):
        fetch, extract, solve, wiki = _patch_layers(
            self, extracted={**FULL_FIELDS, "wire_diameter": (0.53, "cm")})
        with fetch, extract, solve as m_solve, wiki:
            reproduce_s4l._reproduce_s4l(arxiv_id="x", write_wiki=False)
        self.assertAlmostEqual(m_solve.call_args.kwargs["wire_diameter_mm"], 5.3)


class AssumptionsTests(unittest.TestCase):
    def test_assumptions_merged_and_declared(self):
        fetch, extract, solve, wiki = _patch_layers(
            self, extracted={**FULL_FIELDS, "I_peak": (2.0, "A")})
        with fetch, extract, solve as m_solve, wiki:
            r = reproduce_s4l._reproduce_s4l(
                arxiv_id="x", write_wiki=False,
                assumptions={"freq_hz": 3000.0, "current_A": 1.0})
        self.assertEqual(m_solve.call_args.kwargs["freq_hz"], 3000.0)
        self.assertEqual(m_solve.call_args.kwargs["current_A"], 1.0)  # 覆盖抽取值 2.0
        decl = "\n".join(r["declarations"])
        self.assertIn("论文未报告，采用显式假设：freq_hz=3000.0", decl)
        self.assertIn("显式假设覆盖抽取值：current_A=1.0（抽取值=2.0）", decl)
        self.assertIn("declarations", r["report"])

    def test_assumptions_fill_missing_required(self):
        # 论文缺线径，assumptions 显式补上 → 不报错
        extracted = {"wing_diameter": (0.1, "m"), "turns_per_wing": (9, None)}
        fetch, extract, solve, wiki = _patch_layers(self, extracted=extracted)
        with fetch, extract, solve as m_solve, wiki:
            r = reproduce_s4l._reproduce_s4l(
                arxiv_id="x", write_wiki=False,
                assumptions={"wire_diameter_mm": 5.3})
        m_solve.assert_called_once()
        self.assertEqual(m_solve.call_args.kwargs["wire_diameter_mm"], 5.3)
        self.assertIn("论文未报告，采用显式假设：wire_diameter_mm=5.3",
                      "\n".join(r["declarations"]))


class ParamsPathTests(unittest.TestCase):
    def test_params_bypasses_extract(self):
        fetch, extract, solve, wiki = _patch_layers(self)
        with fetch as m_fetch, extract as m_extract, solve as m_solve, wiki:
            r = reproduce_s4l._reproduce_s4l(
                params={"wing_diameter": 0.1, "turns_per_wing": 9,
                        "wire_diameter_mm": 5.3, "freq_hz": 3000.0},
                write_wiki=False)
        m_fetch.assert_not_called()
        m_extract.assert_not_called()
        m_solve.assert_called_once()
        self.assertIsNone(r["extract_duration_s"])
        self.assertEqual(r["status"], "ok")

    def test_params_missing_required(self):
        r = reproduce_s4l._reproduce_s4l(params={"radius": 0.05})
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "extract")
        self.assertTrue(any("turns_per_wing" in m for m in r["missing"]))


class SolveErrorTests(unittest.TestCase):
    def test_l2_error_stage_preserved(self):
        l2_err = {"status": "error", "stage": "run",
                  "stages": {"compile": {"ok": True, "duration_s": 0.1}},
                  "error": "GUI 求解未完成"}
        fetch, extract, solve, wiki = _patch_layers(self, solve_result=l2_err)
        with fetch, extract, solve, wiki as m_wiki:
            r = reproduce_s4l._reproduce_s4l(arxiv_id="x")
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["stage"], "solve")
        self.assertEqual(r["solve_stage"], "run")       # L2 分 stage 细节不丢
        self.assertEqual(r["error"], "GUI 求解未完成")
        m_wiki.assert_not_called()

    def test_failed_verification_passthrough(self):
        l2 = {**SOLVE_OK, "status": "failed_verification",
              "verification": {"verify_mode": "feature", "verdict": "fail"}}
        fetch, extract, solve, wiki = _patch_layers(self, solve_result=l2)
        with fetch, extract, solve, wiki:
            r = reproduce_s4l._reproduce_s4l(arxiv_id="x")
        self.assertEqual(r["status"], "failed_verification")
        self.assertEqual(r["solve_stage"], "verify")


class WikiTests(unittest.TestCase):
    def test_wiki_failure_degrades_to_warning(self):
        fetch, extract, solve, wiki = _patch_layers(self)
        with fetch, extract, solve, \
                patch.object(reproduce_s4l.wiki_tool, "_wiki_write",
                             side_effect=RuntimeError("disk full")):
            r = reproduce_s4l._reproduce_s4l(arxiv_id="x")
        self.assertEqual(r["status"], "ok")             # wiki 失败不阻断
        self.assertTrue(any("wiki" in w for w in r["warnings"]))

    def test_write_wiki_false_skips(self):
        fetch, extract, solve, wiki = _patch_layers(self)
        with fetch, extract, solve, wiki as m_wiki:
            reproduce_s4l._reproduce_s4l(arxiv_id="x", write_wiki=False)
        m_wiki.assert_not_called()


if __name__ == "__main__":
    unittest.main()
