"""ToolDescriptor：在 litmusAgent ToolSpec 之外的元数据层。

新工具 = 新文件 + 模块级 DESCRIPTOR + 注册一行，编排层零改动。
详见 mydocs/specs/2026-07-21_m1-research-agent.md §4.4.1。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    """OpenAI 兼容的 function calling schema（与 litmusAgent ToolSpec 对齐）。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any] = field(repr=False)

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolDescriptor:
    """工具元数据：注册即生效，`requires` 供启动自检，`cost_hint` 供规划参考。"""

    spec: ToolSpec
    category: str  # literature / simulation / compute / knowledge / verify
    cost_hint: str = "cheap"  # free | cheap | expensive
    async_capable: bool = False
    requires: list[str] = field(default_factory=list)
    produces_artifacts: bool = False
