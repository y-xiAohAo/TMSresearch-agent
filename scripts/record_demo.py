#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""录制 research-agent 纯文献链路 demo 并渲染为 GIF。

两段式：
    python scripts/record_demo.py                 # 录制 + 渲染
    python scripts/record_demo.py --record-only   # 只录制 transcript
    python scripts/record_demo.py --render-only   # 复用已有 transcript 渲染 GIF

录制：真实运行 run_research_sync（文献链路：arxiv_search -> paper_analyze -> wiki_write），
tee stdout 到 artifacts/demo_transcript.txt，同时写带时间戳的 sidecar
artifacts/demo_transcript.jsonl（渲染回放节奏用）。

渲染：等宽字体逐行打字机回放，~4x 速度，>2s 空洞停顿截断为 0.8s，
结尾定格 3s，宽度 <=960px，GIF < 5MB（超了抽帧/降色）。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_TXT = ROOT / "artifacts" / "demo_transcript.txt"
TRANSCRIPT_JSONL = ROOT / "artifacts" / "demo_transcript.jsonl"
GIF_PATH = ROOT / "docs" / "images" / "demo.gif"

QUESTION = (
    "请按以下步骤完成一个迷你科研任务（每步一个工具调用，不要重复调用）："
    "1) 用 arxiv_search 检索 arXiv:2511.00744 这篇论文；"
    "2) 用 paper_analyze 抽取它的仿真参数（focus=simulation_params）；"
    "3) 用 wiki_write 把结论记入 wiki（标题含 'TMS 复现'）。"
    "完成后简要总结。"
)

# ---------------------------------------------------------------- 录制


def record() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    sys.path.insert(0, str(ROOT / "src"))
    from research_agent.agent import run_research_sync  # noqa: E402

    TRANSCRIPT_TXT.parent.mkdir(parents=True, exist_ok=True)
    txt_f = open(TRANSCRIPT_TXT, "w", encoding="utf-8")
    jsonl_f = open(TRANSCRIPT_JSONL, "w", encoding="utf-8")

    class Tee(io.TextIOBase):
        def __init__(self, stream):
            self.stream = stream
            self.buf = ""

        def write(self, s):
            self.stream.write(s)
            self.stream.flush()
            txt_f.write(s)
            self.buf += s
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                jsonl_f.write(json.dumps({"t": time.time(), "line": line}, ensure_ascii=False) + "\n")
                jsonl_f.flush()
            return len(s)

        def flush(self):
            self.stream.flush()

    old_out = sys.stdout
    sys.stdout = Tee(old_out)
    t0 = time.time()
    try:
        print("=" * 60)
        print("Research Agent Demo — 纯文献链路 (arxiv_search / paper_analyze / wiki_write)")
        print("=" * 60)
        print(f"[Question] {QUESTION}\n")

        result = run_research_sync(QUESTION)

        if result.get("missing"):
            print(f"[Missing requirements] {result['missing']}\n")
        print("=" * 60)
        print("[Final Answer]")
        print("=" * 60)
        print(result["answer"])
        rc = 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n[ERROR] demo 运行失败: {exc!r}")
        rc = 1
    finally:
        sys.stdout = old_out
        txt_f.close()
        jsonl_f.close()

    print(f"\n[record] 时长 {time.time() - t0:.1f}s -> {TRANSCRIPT_TXT}")
    return rc


# ---------------------------------------------------------------- 渲染

COLS = 80
ROWS = 34
FONT_SIZE = 16
FPS = 10
SPEED = 4.0
PAUSE_CAP = 0.8      # >2s 的真实停顿截断为此时长
PAUSE_THRESH = 2.0
MIN_LINE_TIME = 0.06
SEC_PER_CHAR = 0.02  # 每行最低阅读节奏（CJK 按 2 列计）
MIN_SEG_TIME = 0.2   # 每个折行段的最低展示时长
END_HOLD = 3.0
PADDING = 14
MAX_GIF_BYTES = 5 * 1024 * 1024


def _load_fonts():
    from PIL import ImageFont

    latin = None
    for cand in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf"):
        if Path(cand).exists():
            latin = ImageFont.truetype(cand, FONT_SIZE)
            break
    if latin is None:
        latin = ImageFont.load_default()
    cjk = latin
    for cand in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(cand).exists():
            cjk = ImageFont.truetype(cand, FONT_SIZE)
            break
    return latin, cjk


def _is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F")


def wrap_line(line: str, cols: int = COLS) -> list[str]:
    """按显示宽度（CJK 算 2 列）折行。"""
    out, cur, w = [], "", 0
    for ch in line:
        cw = 2 if _is_wide(ch) else 1
        if w + cw > cols:
            out.append(cur)
            cur, w = "", 0
        cur += ch
        w += cw
    out.append(cur)
    return out or [""]


def load_timed_lines() -> list[tuple[float, str]]:
    """返回 [(在该行显示前停留的秒数, 行文本)]，已应用变速与停顿截断。"""
    entries = []
    with open(TRANSCRIPT_JSONL, encoding="utf-8") as f:
        for raw in f:
            e = json.loads(raw)
            entries.append((e["t"], e["line"]))
    if not entries:
        raise RuntimeError(f"transcript 为空: {TRANSCRIPT_JSONL}")

    timed: list[tuple[float, str]] = []
    prev_t = entries[0][0]
    for t, line in entries:
        gap = (t - prev_t) / SPEED
        if t - prev_t > PAUSE_THRESH:
            gap = PAUSE_CAP
        prev_t = t
        for j, seg in enumerate(wrap_line(line)):
            timed.append((gap if j == 0 else 0.0, seg))
    return timed


def render() -> None:
    from PIL import Image, ImageDraw

    latin, cjk = _load_fonts()
    cell_w = latin.getlength("M")
    cell_h = FONT_SIZE + 6
    width = min(int(cell_w * COLS) + PADDING * 2, 960)
    height = cell_h * ROWS + PADDING * 2

    timed = load_timed_lines()
    # 时间轴：累计每行结束后的展示时刻
    frames: list[tuple[int, list[str]]] = []  # (duration_ms, visible_lines)
    visible: list[str] = []
    for delay, seg in timed:
        visible.append(seg)
        window = visible[-ROWS:]
        # stdout 常成段涌出（时间戳相同），按行长给一个阅读节奏下限
        seg_cols = sum(2 if _is_wide(c) else 1 for c in seg)
        floor = MIN_SEG_TIME + seg_cols * SEC_PER_CHAR
        dur = max(int(delay * 1000), int(floor * 1000), int(MIN_LINE_TIME * 1000))
        frames.append((dur, list(window)))

    # 结尾定格
    frames.append((int(END_HOLD * 1000), list(visible[-ROWS:])))

    def draw(lines: list[str]) -> Image.Image:
        img = Image.new("RGB", (width, height), (12, 12, 12))
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            x = PADDING
            y = PADDING + i * cell_h
            for ch in ln:
                font = cjk if _is_wide(ch) else latin
                d.text((x, y), ch, font=font, fill=(230, 230, 230))
                x += cell_w * (2 if _is_wide(ch) else 1)
        return img

    pil_frames = [draw(ls) for _, ls in frames]
    durations = [d for d, _ in frames]

    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    colors = 128
    while True:
        paletted = [f.convert("P", palette=Image.ADAPTIVE, colors=colors) for f in pil_frames]
        paletted[0].save(
            GIF_PATH, save_all=True, append_images=paletted[1:],
            duration=durations, loop=0, optimize=True,
        )
        size = GIF_PATH.stat().st_size
        if size < MAX_GIF_BYTES and colors <= 128:
            if size < MAX_GIF_BYTES:
                break
        if size < MAX_GIF_BYTES:
            break
        if colors > 32:
            colors //= 2
            print(f"[render] {size/1e6:.1f}MB 超限，降色到 {colors}")
        else:
            # 抽帧：每 2 帧取 1，时长相加
            frames = [(frames[i][0] + (frames[i + 1][0] if i + 1 < len(frames) else 0), frames[i + 1][1] if i + 1 < len(frames) else frames[i][1])
                      for i in range(0, len(frames), 2)]
            pil_frames = [draw(ls) for _, ls in frames]
            durations = [d for d, _ in frames]
            colors = 128
            print(f"[render] 仍超限，抽帧至 {len(frames)} 帧")

    total_s = sum(durations) / 1000
    print(f"[render] {GIF_PATH}  {GIF_PATH.stat().st_size/1e6:.2f}MB  "
          f"{len(durations)} 帧  时长 {total_s:.1f}s  {width}x{height}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--record-only", action="store_true")
    g.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    rc = 0
    if not args.render_only:
        rc = record()
    if rc == 0 and not args.record_only:
        render()
    return rc


if __name__ == "__main__":
    sys.exit(main())
