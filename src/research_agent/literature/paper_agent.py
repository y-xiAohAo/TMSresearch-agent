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
        self._search_count = 0
        self._turn = 0
        self._current_schema: ExtractionSchema | None = None

    # ---------- S1 遥测 ----------

    def _filled_fields(self) -> int:
        """已收集文本中出现非 null 引句字段数的粗估（遥测用，不调 LLM）。"""
        if not self._collected or not self._current_schema:
            return 0
        text = "\n".join(self._collected)
        filled = 0
        for group in self._current_schema.evidence_group_fields:
            for field_key in ("type", "radius_m", "position"):
                # 粗估：组内字段名在文本中出现即算有线索（仅遥测，不影响逻辑）
                if field_key in text:
                    filled += 1
        return filled

    def _required_count(self) -> int:
        """evidence 组 dataclass 的字段总数（遥测分母）。"""
        if not self._current_schema:
            return 0
        import typing
        from dataclasses import fields as dc_fields

        total = 0
        hints = typing.get_type_hints(self._current_schema.entity)
        for g in self._current_schema.evidence_group_fields:
            group_cls = hints.get(g)
            if group_cls is not None:
                total += len(dc_fields(group_cls))
        return total

    def _status_line(self) -> str:
        filled = self._filled_fields()
        req = self._required_count()
        left = max(0, self._cfg.max_turns - self._turn)
        return (
            f"\n\n[Status] Fields={filled}/{req} "
            f"| PagesRead={len(self._pages_read)} "
            f"| Searches={self._search_count}/{self._cfg.max_searches} "
            f"| TurnsLeft={left}"
        )

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
        # S1 已读去重：请求的页全部已读过时直接提示 finish，不重复占位
        requested = {p for p in range(start, end + 1) if p in self._doc.page_texts}
        already = requested & self._pages_read
        if requested and already == requested:
            return {
                "text": f"页 {start}-{end} 已全部读过，无需重读。请用已有内容 finish（或读其它未读页）。",
                "pages_read": sorted(self._pages_read),
                "deduplicated": True,
            }
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
        # S1 search 计数：达上限后拒绝并引导收尾
        if self._search_count >= self._cfg.max_searches:
            return {
                "hits": [],
                "note": f"搜索已达上限({self._cfg.max_searches})。请 read_pages 或用已有内容 finish。",
            }
        self._search_count += 1
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
        self._current_schema = schema
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
        self._self_checked = False
        self._last_progress = -1
        self._stall_count = 0

        for turn in range(1, self._cfg.max_turns + 1):
            self._turn = turn

            # S3a 自检闸：self_check_turn 轮注入一次程序化字段缺口提示
            if turn == self._cfg.self_check_turn and not self._self_checked:
                self._self_checked = True
                gaps = self._field_gaps()
                messages.append({"role": "user", "content": (
                    f"[自检] 尚无引句支撑的字段：{gaps}。剩余 {self._cfg.max_turns - turn} 轮，"
                    "请针对性 read_pages 补，或现在 finish（信息不足可将 sufficient 设为 false）。"
                )})

            # S3b 轮 closing_turn 起工具集收缩为仅 finish（物理上无法继续收集）
            allowed_tools = sub_tools if turn < self._cfg.closing_turn else self._finish_only_schema()
            if turn == self._cfg.closing_turn:
                messages.append({"role": "user", "content": (
                    "[收口模式] 已进入收尾阶段：禁止开启新探索（read/search 已禁用），"
                    "请用已收集的内容立即 finish。"
                )})

            resp = self._llm(messages, allowed_tools)
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
                    verdict, err, result = self._validate_finish(args, schema)
                    if verdict == "reject":
                        messages.append(self._tool_msg(tc, f"校验失败：{err}"))
                        finish_fails += 1
                        last_err = err
                        if finish_fails >= self._cfg.max_finish_fails:
                            return self._done("extraction_failed", started, error=last_err)
                    else:
                        messages.append(self._tool_msg(tc, "accepted"))
                        return self._done(verdict, started, result=result)
                else:
                    out = self._dispatch(name, args)
                    messages.append(self._tool_msg(tc, out))

            # S4 退化检测：连续 max_stall 轮无进度（无新字段线索且无新页）→ 早退强制合成
            progress = self._filled_fields() * 10 + len(self._pages_read)
            if progress == self._last_progress:
                self._stall_count += 1
            else:
                self._stall_count = 0
            self._last_progress = progress
            if (
                self._cfg.force_synthesize
                and self._collected
                and self._stall_count >= self._cfg.max_stall
            ):
                messages.append({"role": "user", "content": (
                    f"[退化检测] 已连续 {self._stall_count} 轮无新进展。停止探索，立即用已有内容 finish。"
                )})
                return self._force_synthesize(messages, schema, started)

        # S3c 强制合成：预算耗尽时用已收集内容强制 best-effort 合成
        if self._cfg.force_synthesize and self._collected:
            return self._force_synthesize(messages, schema, started)
        return self._done("incomplete", started, error=f"达到轮次上限 {self._cfg.max_turns}")

    # ---------- S3 循环控制辅助 ----------

    def _field_gaps(self) -> str:
        """程序化字段缺口：evidence 组内仍全 null 的字段（自检闸用）。"""
        if not self._current_schema:
            return "(未知)"
        import typing

        hints = typing.get_type_hints(self._current_schema.entity)
        collected = "\n".join(self._collected)
        gaps = []
        for g in self._current_schema.evidence_group_fields:
            group_cls = hints.get(g)
            if group_cls is None:
                continue
            from dataclasses import fields as dc_fields

            for f in dc_fields(group_cls):
                if f.name not in collected:
                    gaps.append(f"{g}.{f.name}")
        return ", ".join(gaps) if gaps else "(无，全部有线索)"

    def _finish_only_schema(self) -> list[dict]:
        return [t for t in self._sub_tool_schemas() if t["function"]["name"] == "finish"]

    def _force_synthesize(self, messages: list[dict], schema, started: float) -> dict:
        """预算耗尽时强制 best-effort 合成（Aviary 教训：截断轨迹不能作废）。"""
        messages.append({"role": "user", "content": (
            "预算已耗尽。立即用 finish 提交当前已收集的信息"
            "（信息不足的字段填 null 并将 sufficient 设为 false），不要调用其它工具。"
        )})
        resp = self._llm(messages, self._finish_only_schema())
        tool_calls = resp.get("tool_calls") or []
        for tc in tool_calls:
            if tc["function"]["name"] != "finish":
                continue
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                continue
            verdict, err, result = self._validate_finish(args, schema)
            if verdict != "reject":
                return self._done("best_effort_ok" if verdict == "ok" else verdict, started, result=result)
        return self._done("insufficient", started, error="强制合成未产出可用结果")

    def _dispatch(self, name: str, args: dict) -> str:
        try:
            if name == "get_outline":
                out = json.dumps(self._tool_get_outline(args.get("arxiv_id", "")), ensure_ascii=False)
            elif name == "read_pages":
                out = json.dumps(
                    self._tool_read_pages(args.get("arxiv_id", ""), args.get("start", 1), args.get("end", 3)),
                    ensure_ascii=False,
                )
            elif name == "search_text":
                out = json.dumps(
                    self._tool_search_text(args.get("arxiv_id", ""), args.get("query", "")),
                    ensure_ascii=False,
                )
            else:
                out = f"未知工具：{name}"
            return out + self._status_line()  # S1：统一追加状态遥测行
        except Exception as exc:  # noqa: BLE001
            return f"工具 {name} 执行失败：{exc}"

    def _validate_finish(self, args: dict, schema: ExtractionSchema) -> tuple[str, str, dict | None]:
        """S2 finish 契约校验。返回 (verdict, err, result)。

        finish 新签名：{fields: {...}, confidence: 1-5, sufficient: bool}
        兼容旧外壳 {"result": {...}}。
        verdict: "ok" | "insufficient" | "reject"
        """
        fields = args.get("fields") or args.get("result") or args
        if not isinstance(fields, dict):
            return "reject", "finish 的 fields 必须是 JSON 对象", None

        # 契约字段校验（CORE-Agent 模式：程序化 schema 检查）
        # envelope confidence 仅来自 args（1-5 自评），不与 fields 内 schema 级 confidence(high/medium/low) 混淆
        confidence = args.get("confidence")
        if confidence is not None and (not isinstance(confidence, int) or not (1 <= confidence <= 5)):
            return "reject", f"confidence 必须是 1-5 的整数，实际 {confidence}", None
        sufficient = args.get("sufficient", True)

        # 归一化：缺失的参数组补成空组（引句与字段级 confidence 必须真实存在，不补）
        result = dict(fields)
        for key in schema.required_top_keys:
            if key in ("evidence_quotes", "confidence"):
                continue
            if key not in result or result[key] is None:
                result[key] = {}
        # confidence 双轨：fields 自带 high/medium/low 优先；否则由 envelope 1-5 映射
        if result.get("confidence") not in ("high", "medium", "low"):
            if isinstance(confidence, int):
                result["confidence"] = {1: "low", 2: "low", 3: "medium", 4: "high", 5: "high"}[confidence]
            else:
                result["confidence"] = "low"

        # sufficient=True 时至少要有 1 个组含非 null 字段，否则应报 sufficient=False
        if sufficient:
            has_value = any(
                isinstance(result.get(g), dict) and any(v is not None for v in result[g].values())
                for g in schema.evidence_group_fields
            )
            if not has_value:
                return (
                    "reject",
                    "sufficient=True 但所有参数字段均为 null。若信息不足，请将 sufficient 设为 false 提交。",
                    None,
                )

        # 三道闸（结构/枚举、引句覆盖、引句真实性）
        try:
            validate_payload(result, schema, paper_text=paper_cache.full_text(self._doc.page_texts))
        except ValueError as exc:
            return "reject", str(exc), None

        result["_meta"] = {"confidence": confidence, "sufficient": bool(sufficient)}
        verdict = "ok" if sufficient else "insufficient"
        return verdict, "", result

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
            fn("finish",
               "提交最终结果并结束（必须单独调用）。fields 为按任务 schema 的参数 JSON；"
               "confidence 为 1-5 整数自评；信息不足时设 sufficient=false（合法出口）。",
               {
                   "fields": {"type": "object", "description": "按任务 schema 的参数 JSON"},
                   "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                   "sufficient": {"type": "boolean", "default": True},
               },
               ["fields"]),
        ]
