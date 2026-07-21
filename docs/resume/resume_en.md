# Resume Material — Research Agent (EN)

## Research Agent — LLM-driven Scientific Research Agent (Jul 2026)

**One-liner**: Built a ReAct-style research agent on a self-developed agent framework and DeepSeek, orchestrating six tools — Sim4Life manual RAG, Sim4Life headless simulation scripting, TMS coil optimization, and web search — with experiment tracking and cross-session knowledge memory.

- **Extensible tool orchestration**: Designed a `ToolDescriptor` metadata layer (category / cost hint / requirements / async capability) so tools are effective on registration with startup dependency checks; six tools are exposed to the LLM through OpenAI-compatible function calling, and new tools require zero orchestration changes.
- **Headless automation of commercial software**: Diagnosed and resolved Sim4Life's FlexNet license initialization for GUI-less execution; packaged a "script generation → subprocess execution → artifact collection" modeling tool — headless builds verified 3/3 with ~15.3s average runtime.
- **RAG grounding with anti-hallucination**: Integrated a self-built Sim4Life manual RAG service (hybrid Dense+BM25 retrieval, cross-encoder reranker, UI ground-truth registry, counter-evidence validation); failure paths raise explicit errors instead of silent degradation, preventing hallucinated guidance.
- **Evaluation & memory**: Implemented an after_tool tracking hook logging every tool call (args/result/duration/success) to JSONL — measured 100% success over 30 real calls; built a wiki memory tool and verified cross-session recall (5 prior entries retrieved by a fresh session).

**Stack**: Python, OpenAI Function Calling, ReAct, Prompt Engineering, Agentic RAG, Hybrid Retrieval, Reranker, DeepSeek, FastAPI (SSE), Tavily, headless Sim4Life, NSGA2, pytest, Git.

---

### Short version

Research agent: ReAct loop driving 6 tools (RAG / headless simulation / optimization / search / wiki memory) with 100% measured tool-call success (30/30) and verified cross-session memory recall.
