#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_agent.py：论文理解子代理（轻量有界 ReAct 循环）。

主 agent 调 paper_analyze → 本子代理用 4 个内部工具
（get_outline/read_pages/search_text/finish）语义导航论文，
按 focus 对应 schema 抽取参数，finish 结果强制过三道闸。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from research_agent.config import SETTINGS
from research_agent.literature import paper_cache
from research_agent.literature.client import ArxivClient
from research_agent.literature.pdf import read_pdf_pages
from research_agent.literature.schemas import (
    ExtractionSchema,
    PaperAgentConfig,
    get_schema,
    validate_payload,
)

_SYSTEM_PROMPT = """你是论文结构导航员与参数抽取员。任务：理解一篇论文并抽取 {focus} 对应的结构化参数。

流程建议：
1. 先 get_outline 看论文骨架，再决定 read_pages 读哪些页（优先方法/实验/仿真章节，摘要引言价值低）。
2. 最多读约 {max_segments} 个页段；可用 search_text 辅助定位关键词。
3. 信息足够后用 finish 提交结果 JSON。

硬规则：
- 论文未提及的字段填 null，禁止编造。
- finish 的 JSON 结构：{schema_json}
- 每个非 null 字段必须在 evidence_quotes 给出原文引句。
- finish 必须单独调用（不能和其它工具同一轮）。
- 输出给工具外的任何自由文本都不作数，必须经 finish 提交。"""


@dataclass
class _PaperDoc:
    """已加载论文（全文 + outline + 缓存）。"""

    arxiv_id: str
    version: str
    page_texts: dict[int, str]
    outline: list[dict]

    @property
    def pages(self) -> int:
        return len(self.page_texts)


class PaperUnderstandingAgent:
    """论文理解子代理。"""

    def __init__(
        self,
        llm_chat: Callable[[list[dict], list[dict] | None], dict],
        config: PaperAgentConfig | None = None,
        arxiv_client: ArxivClient | None = None,
    ):
        self._llm = llm_chat
        self._cfg = config or PaperAgentConfig()
        self._client = arxiv_client or ArxivClient()
        self._doc: _PaperDoc | None = None
        self._trace: list[dict] = []
        self._pages_read: set[int] = set()
        self._collected: list[str] = []  # [(page_range, text)] 仅用于上下文与 provenance

    # ---------- 内部工具 ----------

    def _tool_get_outline(self, arxiv_id: str) -> dict:
        self._ensure_doc(arxiv_id)
        return {"pages": self._doc.pages, "outline": self._doc.outline}

    def _tool_read_pages(self, arxiv_id: str, start: int, end: int) -> dict:
        self._ensure_doc(arxiv_id)
        start = max(1, int(start))
        end = min(self._doc.pages, int(end))
        if start > end:
            return {"error": f"页码区间非法：{start}-{end}（共 {self._doc.pages} 页）"}
        texts = []
        for p in range(start, end + 1):
            if p in self._doc.page_texts:
                texts.append(f"--- Page {p} ---\n{self._doc.page_texts[p]}")
                self._pages_read.add(p)
        text = "\n".join(texts)
        cap = self._cfg.read_segment_chars
        if len(text) > cap:
            text = text[:cap] + "\n...[本段已截断]"
        self._collected.append(f"[pages {start}-{end}]\n{text}")
        self._enforce_collected_cap()
        return {"text": text, "pages_read": sorted(self._pages_read)}

    def _tool_search_text(self, arxiv_id: str, query: str) -> dict:
        self._ensure_doc(arxiv_id)
        q = query.strip().lower()
        if not q:
            return {"hits": []}
        hits = []
        for p in sorted(self._doc.page_texts):
            idx = self._doc.page_texts[p].lower().find(q)
            if idx >= 0:
                snippet = self._doc.page_texts[p][max(0, idx - 60): idx + 100].replace("\n", " ")
                hits.append({"page": p, "snippet": snippet})
            if len(hits) >= 10:
                break
        return {"hits": hits}

    # ---------- 加载 ----------

    def _ensure_doc(self, arxiv_id: str) -> None:
        if self._doc and self._doc.arxiv_id == arxiv_id:
            return
        from research_agent.literature.client import normalize_arxiv_id

        norm_id, version = normalize_arxiv_id(arxiv_id)
        cached = paper_cache.load(norm_id, version)
        if cached:
            page_texts = {int(k): v for k, v in cached["page_texts"].items()}
        else:
            pdf_path = self._client.download_pdf(norm_id)
            page_texts = self._extract_pages(str(pdf_path))
            paper_cache.save(norm_id, version, {
                "pages": len(page_texts),
                "page_texts": {str(k): v for k, v in page_texts.items()},
            })
        outline = paper_cache.build_outline(page_texts, self._cfg.outline_chars)
        self._doc = _PaperDoc(arxiv_id=norm_id, version=version, page_texts=page_texts, outline=outline)

    @staticmethod
    def _extract_pages(pdf_path: str) -> dict[int, str]:
        import fitz

        texts: dict[int, str] = {}
        with fitz.open(pdf_path) as doc:
            for i in range(doc.page_count):
                texts[i + 1] = doc.load_page(i).get_text("text")
        return texts

    def _enforce_collected_cap(self) -> None:
        total = sum(len(t) for t in self._collected)
        while total > self._cfg.collected_cap_chars and len(self._collected) > 1:
            self._collected.pop(0)
            total = sum(len(t) for t in self._collected)

    # ---------- 主循环 ----------

    def run(self, arxiv_id: str, focus: str = "simulation_params") -> dict:
        started = time.perf_counter()
        schema = get_schema(focus)
        if schema is None:
            return {"status": "error", "error": f"unsupported focus: {focus}"}
        try:
            self._ensure_doc(arxiv_id)
        except Exception as exc:
            return {"status": "network_unavailable", "error": str(exc)}
        if self._doc.pages == 0:
            return {"status": "not_found", "arxiv_id": arxiv_id}

        model = self._cfg.model or SETTINGS.chat_model
        system = _SYSTEM_PROMPT.format(
            focus=focus,
            max_segments=self._cfg.max_page_segments,
            schema_json=schema.prompt_template.split("<external_document>")[0].split("输出 JSON 结构：")[-1].strip()
            if "{schema_json}" not in schema.prompt_template
            else schema.prompt_template.format(text="", schema_json="...").split("<external_document>")[0],
        )
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"分析论文 {arxiv_id}，目标 focus={focus}。先用 get_outline 了解结构。"},
        ]
        sub_tools = self._sub_tool_schemas()
        finish_fails = 0
        nudges = 0
        last_err = ""

        for turn in range(1, self._cfg.max_turns + 1):
            resp = self._llm(messages, sub_tools)
            # 修正 P0-1：assistant 消息必须入列
            messages.append(resp.get("message") or {"role": "assistant", "content": resp.get("content", "")})

            tool_calls = resp.get("tool_calls") or []
            if not tool_calls:
                nudges += 1
                hint = (
                    "请用 finish 工具提交结果 JSON。"
                    if self._collected
                    else "尚未阅读论文内容，请先用 get_outline / read_pages 阅读。"
                )
                messages.append({"role": "user", "content": hint})
                if nudges >= self._cfg.max_nudges:
                    return self._done("incomplete", started, error="LLM 未调用工具")
                continue

            # finish 必须独占一轮
            finish_calls = [tc for tc in tool_calls if tc["function"]["name"] == "finish"]
            if finish_calls and len(tool_calls) > 1:
                for tc in tool_calls:
                    messages.append(self._tool_msg(tc, "finish 必须独占一轮，请单独调用。"))
                continue

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    messages.append(self._tool_msg(tc, "arguments 不是合法 JSON"))
                    continue
                self._trace.append({"turn": turn, "tool": name, "args": args})

                if name == "finish":
                    ok, err, result = self._validate_finish(args, schema)
                    messages.append(self._tool_msg(tc, "accepted" if ok else f"校验失败：{err}"))
                    if ok:
                        return self._done("ok", started, result=result)
                    finish_fails += 1
                    last_err = err
                    if finish_fails >= self._cfg.max_finish_fails:
                        return self._done("extraction_failed", started, error=last_err)
                else:
                    out = self._dispatch(name, args)
                    messages.append(self._tool_msg(tc, out))

        return self._done("incomplete", started, error=f"达到轮次上限 {self._cfg.max_turns}")

    # ---------- 辅助 ----------

    def _dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "get_outline":
                return json.dumps(self._tool_get_outline(args.get("arxiv_id", "")), ensure_ascii=False)
            if name == "read_pages":
                return json.dumps(
                    self._tool_read_pages(args.get("arxiv_id", ""), args.get("start", 1), args.get("end", 3)),
                    ensure_ascii=False,
                )
            if name == "search_text":
                return json.dumps(
                    self._tool_search_text(args.get("arxiv_id", ""), args.get("query", "")),
                    ensure_ascii=False,
                )
            return f"未知工具：{name}"
        except Exception as exc:  # noqa: BLE001
            return f"工具 {name} 执行失败：{exc}"

    def _validate_finish(self, args: dict, schema: ExtractionSchema) -> tuple[bool, str, dict | None]:
        # finish 的 arguments 是 {"result": {...}} 外壳，校验与返回都取内层 result
        result = args.get("result", args)
        if not isinstance(result, dict):
            return False, "finish 的 result 必须是 JSON 对象", None
        # 归一化：缺失的参数组补成空组（"缺失组 = 全 null"语义等价，不该判失败）
        result = dict(result)
        for key in schema.required_top_keys:
            if key in ("evidence_quotes", "confidence"):
                continue  # 引句与置信度必须真实存在，不补
            if key not in result or result[key] is None:
                result[key] = {}
        try:
            validate_payload(result, schema, paper_text=paper_cache.full_text(self._doc.page_texts))
            return True, "", result
        except ValueError as exc:
            return False, str(exc), None

    @staticmethod
    def _tool_msg(tc: dict, content: str) -> dict:
        return {"role": "tool", "tool_call_id": tc["id"], "content": str(content)[:4000]}

    def _done(self, status: str, started: float, result: dict | None = None, error: str = "") -> dict:
        out = {
            "status": status,
            "result": result,
            "provenance": {
                "pages_read": sorted(self._pages_read),
                "tool_trace": self._trace,
            },
            "duration_s": round(time.perf_counter() - started, 1),
        }
        if error:
            out["error"] = error
        if status in ("incomplete", "extraction_failed"):
            out["partial"] = {
                "collected_pages": sorted(self._pages_read),
                "last_error": error,
                "note": "draft_result 仅供调试，主 agent 不得当合法结果使用",
            }
        return out

    def _sub_tool_schemas(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            }

        pid = {"type": "string", "description": "arxiv id"}
        return [
            fn("get_outline", "获取论文页数与各页标题线索（先看这个了解结构）",
               {"arxiv_id": pid}, ["arxiv_id"]),
            fn("read_pages", "按页阅读论文原文",
               {"arxiv_id": pid,
                "start": {"type": "integer"}, "end": {"type": "integer"}},
               ["arxiv_id", "start", "end"]),
            fn("search_text", "全文搜索关键词，返回命中页码",
               {"arxiv_id": pid, "query": {"type": "string"}}, ["arxiv_id", "query"]),
            fn("finish", "提交最终抽取结果 JSON 并结束（必须单独调用）",
               {"result": {"type": "object", "description": "按任务 schema 的结果 JSON"}},
               ["result"]),
        ]
