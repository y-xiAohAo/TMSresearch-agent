#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1 端到端验证：一个真实科研问题走完整 ReAct 循环。

用法：
    python scripts/demo_research.py
    python scripts/demo_research.py --question "你的问题"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，agent 输出含 emoji/全角字符时需 UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_agent.agent import run_research_sync  # noqa: E402

DEFAULT_QUESTION = (
    "请按以下步骤完成一个迷你科研任务（每步一个工具调用，不要重复调用）："
    "1) 用 sim4life_manual_qa 问：Sim4Life 中创建实心球模型的步骤；"
    "2) 用 tms_optimize 以默认参数跑一次小优化；"
    "3) 用 s4l_write_script 写一个创建实心球并保存的脚本（严格按系统提示中的最小模板，"
    "document.SaveAs 路径用 artifacts 目录下的绝对路径），再用 s4l_run_script 执行；"
    "4) 用 wiki_write 把本次过程与结论写入 wiki（标题含 MRI 线圈）。"
    "完成后简要总结。"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Research Agent M1 demo")
    parser.add_argument("--question", "-q", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print("=" * 60)
    print("Research Agent M1 Demo")
    print("=" * 60)
    print(f"[Question] {args.question}\n")

    result = run_research_sync(args.question)

    if result["missing"]:
        print(f"[Missing requirements] {result['missing']}\n")
    print("=" * 60)
    print("[Final Answer]")
    print("=" * 60)
    print(result["answer"])

    wiki_dir = Path(__file__).resolve().parents[1] / "wiki"
    entries = sorted(wiki_dir.glob("*.md"))
    print("\n" + "=" * 60)
    print(f"[Wiki entries] {len(entries)}")
    for p in entries[-3:]:
        print(f"  - {p.name}")
    if not entries:
        print("  (未产出 wiki 条目)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
