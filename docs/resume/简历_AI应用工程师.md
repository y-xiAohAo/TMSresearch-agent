# 简历素材 — AI 应用工程师向

> 侧重点：工程落地能力（架构、测试、排障、可靠性）。每条可直接粘贴。

## 项目：科研智能体 Research Agent（2026.07）

**一句话**：基于自研 Agent 框架与 DeepSeek 构建 ReAct 科研智能体，编排 Sim4Life 仿真建模、TMS 线圈优化、手册 RAG 与联网检索 6 个工具，具备实验追踪与跨会话记忆能力。

- **设计可扩展的工具编排框架**：定义 `ToolDescriptor` 元数据层（category/cost_hint/requires/async_capable），实现工具"注册即生效、目录式发现、启动时依赖自检"，6 个工具（文献检索/RAG 问答/脚本生成/headless 仿真/科学计算/知识读写）统一经 OpenAI Function Calling 接入，新增工具零编排改动。
- **落地商业软件 headless 自动化**：排查并解决 Sim4Life（FlexNet 许可）无界面运行的许可证初始化问题，封装"脚本生成 → 子进程执行 → 产物回收"的建模工具，headless 建模仿真实测 3/3 成功、平均 15.3s。
- **构建 experiment tracking 实证体系**：实现 after_tool 钩子将每次工具调用（参数/结果/耗时/成败）落盘 JSONL，据此实测工具调用成功率 100%（30/30），并定位出 LLM 长连接挂起这一外部稳定性问题。
- **建立分层测试与鲁棒性保障**：17 个测试覆盖工具契约、故障路径（RAG 宕机明确报错不幻觉、TMS 超时返回结构化 partial、脚本错误回传 stderr）与跨会话记忆召回，端到端 demo 一次跑通（EXIT=0）并产出结构化 wiki 研究记录。

**技术栈**：Python、OpenAI Function Calling、ReAct、FastAPI(SSE)、Tavily API、Chroma RAG、headless Sim4Life、NSGA2 优化、pytest、Git。

---

## 备用短版（一句话 + 2 条）

科研智能体：ReAct 循环驱动 6 工具（RAG/仿真/优化/检索/记忆），工具调用成功率实测 100%。

- 设计 ToolDescriptor 元数据 + 目录式工具框架，6 工具经 Function Calling 统一编排，启动时依赖自检。
- 打通 Sim4Life headless 建模与 TMS 优化链路，experiment tracking 实测 30 次工具调用零失败，跨会话 wiki 记忆可召回。
