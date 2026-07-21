"""research-agent 配置加载：统一从 .env / 环境变量读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    deepseek_base_url: str
    chat_model: str
    tavily_api_key: str
    rag_base_url: str
    s4l_home: str
    s4l_python: str
    tms_project_dir: str
    tms_python: str
    artifacts_dir: str


def _resolve_artifacts_dir() -> str:
    """artifacts 目录锚定到项目根（相对路径时），保证子进程跨 cwd 可用。"""
    raw = os.getenv("ARTIFACTS_DIR", "artifacts").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path)


def load_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        chat_model=os.getenv("CHAT_MODEL", "deepseek-v4-flash").strip(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        rag_base_url=os.getenv("RAG_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        s4l_home=os.getenv("S4L_HOME", "").strip(),
        s4l_python=os.getenv("S4L_PYTHON", "").strip(),
        tms_project_dir=os.getenv("TMS_PROJECT_DIR", "").strip(),
        tms_python=os.getenv("TMS_PYTHON", "").strip(),
        artifacts_dir=_resolve_artifacts_dir(),
    )


SETTINGS = load_settings()
