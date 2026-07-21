#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""experiment tracking：after_tool 钩子，把每次工具调用记录到 JSONL。

每行一条：{ts, tool, args, success, duration_s, error}
这是 spec §4.4.3 预留钩子链的第一个落地实现。
"""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime
from pathlib import Path

from research_agent.config import SETTINGS

_LOG_FILE = "experiment_log.jsonl"


def _log_path() -> Path:
    artifacts = Path(SETTINGS.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts / _LOG_FILE


def track_tool(fn):
    """装饰器：记录工具调用的参数/结果/耗时到 experiment_log.jsonl。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        success = True
        error = ""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            duration = round(time.perf_counter() - started, 3)
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "tool": getattr(fn, "__name__", "unknown"),
                "args": {k: str(v)[:200] for k, v in kwargs.items()},
                "success": success,
                "duration_s": duration,
                "error": error[:500],
            }
            try:
                with open(_log_path(), "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError:
                pass  # tracking 失败不影响工具本身

    return wrapper
