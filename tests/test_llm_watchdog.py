#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 看门狗测试：超时治理——显式超时/重试边界 + 结构化错误（不裸抛、不挂死）。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import httpx
import openai


def _timeout_exc() -> openai.APITimeoutError:
    req = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    return openai.APITimeoutError(request=req)


class LlmClientWatchdogTests(unittest.TestCase):
    def test_make_llm_client_has_explicit_timeout_and_retries(self):
        from research_agent.llm_util import make_llm_client

        client = make_llm_client()
        # 不再裸用 SDK 默认（timeout=600s / max_retries=2 → 最坏挂起 ~30min）
        self.assertLessEqual(client.timeout, 60.0)
        self.assertLessEqual(client.max_retries, 1)

    def test_lit_extract_params_timeout_returns_structured_error(self):
        from research_agent.tools import lit_extract_params

        fake_paper = MagicMock(abstract="A TMS coil study.", title="t")
        with patch.object(lit_extract_params, "ArxivClient") as mock_client_cls, patch.object(
            lit_extract_params, "read_pdf_pages", side_effect=RuntimeError("no pdf")
        ), patch.object(
            lit_extract_params, "extract_sim_params", side_effect=_timeout_exc()
        ):
            mock_client_cls.return_value.fetch.return_value = fake_paper
            mock_client_cls.return_value.download_pdf.side_effect = RuntimeError("no pdf")
            result = lit_extract_params._lit_extract_params("2511.00744")
        self.assertEqual(result["status"], "llm_unavailable")
        self.assertIn("error", result)

    def test_paper_analyze_timeout_returns_structured_error(self):
        from research_agent.tools import paper_analyze

        with patch.object(paper_analyze, "PaperUnderstandingAgent") as mock_agent_cls:
            mock_agent_cls.return_value.run.side_effect = _timeout_exc()
            result = paper_analyze._paper_analyze("2511.00744")
        self.assertEqual(result["status"], "llm_unavailable")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
