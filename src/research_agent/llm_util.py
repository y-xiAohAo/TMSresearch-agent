#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 客户端集中构造点：统一超时/重试治理。

背景：paper_analyze / lit_extract_params 曾各自裸构造 OpenAI()，
SDK 默认 timeout=600s × 默认 2 次重试 = 最坏挂起 ~30 分钟
（M1/resume-packaging 实测 50s~>300s 挂起的根因）。
"""

from __future__ import annotations

from openai import OpenAI

from research_agent.config import SETTINGS

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 1


def make_llm_client(
    timeout: float = DEFAULT_TIMEOUT_S,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAI:
    """构造带显式超时/重试的 DeepSeek（OpenAI 兼容）客户端。"""
    return OpenAI(
        api_key=SETTINGS.deepseek_api_key,
        base_url=SETTINGS.deepseek_base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
