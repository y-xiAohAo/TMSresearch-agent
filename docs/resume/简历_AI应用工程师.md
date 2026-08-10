# 简历素材 — AI 应用工程师向

> 侧重点：工程落地能力（架构、测试、排障、可靠性）。每条可直接粘贴。

## 项目：科研智能体 Research Agent（2026.07–08）

**一句话**：基于自研 Agent 框架与 DeepSeek 构建 ReAct 科研智能体，编排 Sim4Life 仿真建模/有损求解、TMS 线圈优化、手册 RAG 与联网检索工具链，打通"论文→参数→头模仿真→定量场验证"闭环，具备实验追踪与跨会话记忆能力。

- **设计可扩展的工具编排框架**：定义 `ToolDescriptor` 元数据层（category/cost_hint/requires/async_capable），实现工具"注册即生效、目录式发现、启动时依赖自检"，文献检索/RAG 问答/脚本生成/headless 仿真/科学计算/知识读写等工具统一经 OpenAI Function Calling 接入，新增工具零编排改动。
- **落地商业软件 headless 自动化与全自动求解**：逆向 Sim4Life 学生版云端会话 license 机制（headless 无求解许可），落地 `GUI --run` 执行器模式（日志写文件/进程树超时清理/单座串行纪律），实现零人工干预的有损求解与结果提取（h5py 直读 Output.h5，绕过后处理 API）。
- **深度调试与根因分析**：37 轮受控探针定位头模体素化失败的三层叠加根因（网格不纳入实体/默认体素引擎不兼容脚本实体/文档默认毫米致几何小千倍），实践"受控单变量 + 读回验证 + 已知正确模型 A/B 对照"方法论；顺带修复潜伏一整期的单位 bug。
- **构建 experiment tracking 实证体系**：实现 after_tool 钩子将每次工具调用（参数/结果/耗时/成败）落盘 JSONL，据此实测工具调用成功率 100%（30/30），并定位出 LLM 长连接挂起这一外部稳定性问题。
- **可靠性工程**：求解编排设计单文件备份 + mtime 差分定位 + 分 stage 结构化错误；一次磁盘事故（备份全量复制 71G）后复盘修复策略（单文件级 + 自动清理）并补回归测试。
- **建立分层测试与鲁棒性保障**：140+ 测试覆盖工具契约、故障路径（RAG 宕机明确报错不幻觉、TMS 超时返回结构化 partial）与跨会话记忆召回；真实冒烟分级（headless 体素化 25s / GUI 求解 420s），B2 路径行为零回归。

**技术栈**：Python、OpenAI Function Calling、ReAct、FastAPI(SSE)、Tavily API、Chroma RAG、headless/GUI 双模 Sim4Life 自动化、MQS 有损电磁求解、h5py 场提取、NSGA2 优化、pytest、Git。

---

## 备用短版（一句话 + 2 条）

科研智能体：ReAct 循环驱动 6 工具（RAG/仿真/优化/检索/记忆），工具调用成功率实测 100%。

- 设计 ToolDescriptor 元数据 + 目录式工具框架，6 工具经 Function Calling 统一编排，启动时依赖自检。
- 打通 Sim4Life headless 建模与 TMS 优化链路，experiment tracking 实测 30 次工具调用零失败，跨会话 wiki 记忆可召回。
