#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多场景端到端验证：3 类科研场景 × 2 次，统计成功率并生成报告。

场景：
  A 手册问答型：sim4life_manual_qa + wiki_write
  B 优化计算型：tms_optimize + wiki_write
  C 全链路建模型：manual_qa → s4l_write_script → s4l_run_script → wiki_write

输出：artifacts/scenario_report.md + 终端摘要。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from research_agent.agent import run_research_sync  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    {
        "id": "A_manual_qa",
        "title": "手册问答型",
        "question": (
            "用 sim4life_manual_qa 问：Sim4Life 中如何创建一个新的仿真项目？"
            "然后用 wiki_write 把答案要点写入 wiki（标题含 Sim4Life 项目创建）。"
            "两步完成，不要调用其他工具。"
        ),
        "expect_tools": ["sim4life_manual_qa", "wiki_write"],
    },
    {
        "id": "B_optimize",
        "title": "优化计算型",
        "question": (
            "用 tms_optimize 以默认参数跑一次 TMS 线圈优化，"
            "然后用 wiki_write 把关键结果（解数、耗时）写入 wiki（标题含 TMS 优化）。"
            "两步完成，不要调用其他工具。"
        ),
        "expect_tools": ["tms_optimize", "wiki_write"],
    },
    {
        "id": "C_full_pipeline",
        "title": "全链路建模型",
        "question": (
            "完成迷你建模任务：1) 用 sim4life_manual_qa 问创建实心球的步骤；"
            "2) 用 s4l_write_script 按系统提示的最小模板写建球脚本"
            "（document.SaveAs 用 artifacts 目录绝对路径）；"
            "3) 用 s4l_run_script 执行；"
            "4) 用 wiki_write 记录过程（标题含实心球建模）。"
            "每步一次调用，不要重复。"
        ),
        "expect_tools": ["sim4life_manual_qa", "s4l_write_script", "s4l_run_script", "wiki_write"],
    },
]

RUNS_PER_SCENARIO = 2


def _tools_used_in_log(since_ts: str) -> list[str]:
    log = ROOT / "artifacts" / "experiment_log.jsonl"
    if not log.is_file():
        return []
    tools = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ts", "") >= since_ts:
            tools.append(rec.get("tool", ""))
    return tools


def main() -> int:
    print("=" * 60)
    print("Research Agent 多场景验证")
    print("=" * 60)

    results = []
    for sc in SCENARIOS:
        for run_idx in range(1, RUNS_PER_SCENARIO + 1):
            started_mark = datetime.now().isoformat(timespec="seconds")
            started = time.perf_counter()
            print(f"\n[{sc['id']} run{run_idx}] {sc['title']} ...")
            try:
                out = run_research_sync(sc["question"])
                answer = out["answer"] or ""
                err = None
            except Exception as exc:  # noqa: BLE001
                answer = ""
                err = f"{type(exc).__name__}: {exc}"
            duration = round(time.perf_counter() - started, 1)

            used = _tools_used_in_log(started_mark)
            hit = [t for t in sc["expect_tools"] if any(t in u for u in used)]
            ok = err is None and bool(answer.strip()) and len(hit) == len(sc["expect_tools"])
            results.append({
                "scenario": sc["id"],
                "title": sc["title"],
                "run": run_idx,
                "ok": ok,
                "duration_s": duration,
                "tools_hit": hit,
                "tools_expected": sc["expect_tools"],
                "error": err,
            })
            print(f"  -> {'PASS' if ok else 'FAIL'} ({duration}s) tools={hit}")

    total = len(results)
    passed = sum(1 for r in results if r["ok"])
    lines = [
        "# Research Agent 多场景验证报告",
        "",
        f"- 时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 场景数：{len(SCENARIOS)}，每场景 {RUNS_PER_SCENARIO} 次，共 {total} 次",
        f"- 通过：{passed}/{total}（{passed/total:.0%}）",
        "",
        "| 场景 | 次数 | 结果 | 耗时(s) | 命中工具 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scenario']} | run{r['run']} | {'PASS' if r['ok'] else 'FAIL'} "
            f"| {r['duration_s']} | {', '.join(r['tools_hit'])} |"
        )
    report = "\n".join(lines) + "\n"
    report_path = ROOT / "artifacts" / "scenario_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"通过 {passed}/{total}，报告：{report_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
