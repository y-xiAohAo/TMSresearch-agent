# Research Agent

科研智能体：ReAct 循环驱动 Sim4Life 知识库、Sim4Life 建模脚本、TMS 线圈优化与联网搜索，并把研究结论沉淀为个人 wiki 知识库。

## 架构

```
User question
    │
    ▼
┌────────────────────────────────────────────┐
│  ReAct Loop (litmusAgent engine + DeepSeek) │
│  Recall → Ground → Plan → Act → Observe     │
│  → Reflect → Distill                        │
└──┬───────┬────────┬────────┬────────┬──────┘
   │       │        │        │        │
 web_search sim4life s4l_*   tms_    wiki_*
 (Tavily)  _manual_qa (headless optimize (记忆)
           (RAG API) s4l_v1) (NSGA2)
```

- **编排**：litmusAgent `Agent` 引擎（`../litmusAgent`，editable 安装），LLM 走 DeepSeek（OpenAI 兼容）。
- **工具**：`src/research_agent/tools/`，每个文件导出 `ToolDescriptor`（含 category/cost_hint/requires 元数据），`register_all_tools()` 注册。新增工具只需丢一个文件进目录。
- **扩展性**：`ToolDescriptor.requires` 启动自检；M2 将加入目录自动发现、执行钩子链（experiment tracking）与异步任务抽象（见 `mydocs/specs/2026-07-21_m1-research-agent.md` §4.4）。

## 工具清单

| 工具 | 类别 | 说明 |
|---|---|---|
| `web_search` | literature | Tavily 联网检索（泛网页/资讯） |
| `arxiv_search` | literature | arXiv 精确文献检索（分类/日期/作者） |
| `arxiv_fetch` | literature | arXiv 元数据获取 + PDF 归档 |
| `arxiv_read_pdf` | literature | arXiv PDF 按页阅读（pymupdf） |
| `lit_extract_params` | literature | 论文→仿真参数桥（LLM 抽取+引句） |
| `sim4life_manual_qa` | knowledge | Sim4Life 手册 RAG（需先启动 RAG 服务） |
| `s4l_write_script` | simulation | 写 s4l_v1 脚本（自动加 headless 引导头） |
| `s4l_run_script` | simulation | headless 执行 Sim4Life 建模/仿真脚本 |
| `tms_optimize` | compute | TMS 流函数线圈优化（NSGA2 小参数模板） |
| `wiki_write` / `wiki_search` | knowledge | 个人 wiki 记忆读写 |

## 快速开始

```powershell
# 1. 安装（Python 3.12）
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pip install -e ../litmusAgent

# 2. 配置 .env（复制 .env.example，填入 DeepSeek / Tavily key 与路径）

# 3. 启动 RAG 服务（另一个终端）
cd ..\Sim4Life-RAG-Helper
python -m uvicorn api_fastapi:app --host 127.0.0.1 --port 8000

# 4. 测试
python -m pytest tests/

# 5. 端到端 demo
python scripts/demo_research.py
```

## 环境变量

见 `.env.example`：`DEEPSEEK_API_KEY`、`TAVILY_API_KEY`、`RAG_BASE_URL`、`S4L_HOME`、`S4L_PYTHON`、`TMS_PROJECT_DIR`、`TMS_PYTHON`。

## 兄弟项目（只读调用，不改动）

- `../Sim4Life-RAG-Helper`：Sim4Life 手册 RAG 服务
- `../StreamFunctionTMS`：TMS 流函数优化器
- `../litmusAgent`：Agent 引擎
