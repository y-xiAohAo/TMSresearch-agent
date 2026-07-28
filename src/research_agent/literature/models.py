#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""literature 实体定义（M2c）。

结构借鉴：PaperQA Doc/DocDetails 两级分离、Zotero note/attachment 分离、
STORM 证据池编号引用、Elicit 列抽取+原文引句。详见 mydocs/specs/2026-07-22_m2c-literature-layer.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LiteratureItem:
    """轻量文献条目（arxiv_search 返回）。"""

    arxiv_id: str            # 归一化无版本尾，如 "2511.00744"
    version: str             # "v1" / "v2"
    title: str
    authors: list[str]
    published: str           # ISO 日期
    updated: str
    categories: list[str]
    abs_url: str
    ids: dict[str, str] = field(default_factory=dict)
    citation_count: int | None = None    # arXiv API 不提供，预留
    is_retracted: bool | None = None     # 预留（M2c+ 接 Crossref）


@dataclass
class PaperDetails(LiteratureItem):
    """富元数据（arxiv_fetch 时水合）。"""

    abstract: str = ""
    pdf_url: str = ""


@dataclass
class CoilGeometry:
    type: str = ""                       # "loop" | "figure8" | "custom"
    radius_m: float | None = None
    position: str | None = None


@dataclass
class TargetField:
    strength_T: float | None = None
    focal_depth_m: float | None = None
    region: str | None = None


@dataclass
class SimSettings:
    solver: str | None = None
    mesh_cells: int | None = None
    boundary: str | None = None


@dataclass
class AlgoSettings:
    name: str | None = None
    pop_size: int | None = None
    n_gen: int | None = None
    objectives: list[str] | None = None


@dataclass
class SimParamExtraction:
    """创新点 1：论文 -> 仿真参数桥。None 表示论文未提及（禁止编造）。"""

    coil_geometry: CoilGeometry = field(default_factory=CoilGeometry)
    target_field: TargetField = field(default_factory=TargetField)
    simulation: SimSettings = field(default_factory=SimSettings)
    algorithm: AlgoSettings = field(default_factory=AlgoSettings)
    evidence_quotes: dict[str, str] = field(default_factory=dict)  # 字段路径 -> 原文引句
    confidence: str = "low"                # "high" | "medium" | "low"


@dataclass
class IngestNote:
    """创新点 2：文献笔记（Zotero 式 note/attachment 分离）。"""

    arxiv_id: str
    title: str
    abstract_excerpt: str
    agent_notes: str
    evidence_pool: list[str] = field(default_factory=list)   # STORM 式编号引句 [i]
    verification: dict = field(default_factory=dict)         # {claim, status, sim_ref}
    pdf_ref: str | None = None
