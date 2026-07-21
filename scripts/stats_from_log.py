#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 experiment_log.jsonl 生成工具使用统计（成功率/耗时分布）。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research_agent.config import SETTINGS  # noqa: E402


def main() -> int:
    log = Path(SETTINGS.artifacts_dir) / "experiment_log.jsonl"
    if not log.is_file():
        print("无 experiment_log.jsonl")
        return 1

    records = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    by_tool: dict[str, list] = defaultdict(list)
    for r in records:
        by_tool[r.get("tool", "?")].append(r)

    print(f"总调用次数: {len(records)}\n")
    print(f"{'工具':<24}{'次数':>5}{'成功率':>8}{'均耗时s':>9}{'最大s':>8}")
    print("-" * 56)
    total_ok = 0
    for tool, rs in sorted(by_tool.items(), key=lambda x: -len(x[1])):
        n = len(rs)
        ok = sum(1 for r in rs if r.get("success"))
        total_ok += ok
        durations = [r.get("duration_s", 0.0) for r in rs]
        avg = sum(durations) / n
        print(f"{tool:<24}{n:>5}{ok/n:>8.0%}{avg:>9.1f}{max(durations):>8.1f}")
    print("-" * 56)
    print(f"{'总计':<24}{len(records):>5}{total_ok/len(records):>8.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
