# TimeCampus Agent

TimeCampus Agent 是“时光航迹”的独立 Python/LangGraph 子模块，用于后台运营维护、RAG 检索、草案生成、MCP 工具发现和游客导引。Agent 编排刻意放在 Portal 之外，前端保持纯 UI 客户端，Agent 通过 Backend REST API 和 `/mcp` 与系统交互。

当前模块版本：`0.3.0-beta`。

## 环境准备

```powershell
uv sync
Copy-Item .env.example .env
```

填写 `.env`：

```env
TIMECAMPUS_API_BASE_URL=http://127.0.0.1:8080/api/v1
TIMECAMPUS_ADMIN_USERNAME=admin
TIMECAMPUS_ADMIN_PASSWORD=123456
# TIMECAMPUS_ADMIN_TOKEN=
TIMECAMPUS_CHAT_BASE_URL=https://api.deepseek.com/v1
TIMECAMPUS_CHAT_MODEL=deepseek-chat
# TIMECAMPUS_CHAT_API_KEY=
TIMECAMPUS_MCP_URL=http://127.0.0.1:8080/mcp
# TIMECAMPUS_MCP_TOKEN=
TIMECAMPUS_AGENT_API_TOKEN=<backend-and-agent-shared-token>
TIMECAMPUS_AGENT_API_HOST=127.0.0.1
TIMECAMPUS_AGENT_API_PORT=8090
TIMECAMPUS_AGENT_MEMORY_DIR=data/agent-memory
TIMECAMPUS_AGENT_SESSION_HISTORY_LIMIT=40
# TIMECAMPUS_EVAL_LLM_ENABLED=false
```

`TIMECAMPUS_ADMIN_TOKEN` 优先于用户名密码；调用管理端草案接口时必须有管理员 token 或可登录凭据。`rag-search` 在配置 `TIMECAMPUS_MCP_TOKEN` 时优先走 MCP RAG 工具，避免生产 cap 登录流程影响 CLI。

## 运行

先启动 Backend API/MCP：

```powershell
cd ..\TimeCampus
.\tools\start-backend-mcp.ps1
```

再运行 Agent：

```powershell
cd TimeCampus-Agent
uv run timecampus-agent rag-search "主楼旧照"
uv run timecampus-agent draft "为主楼补充面向游客的简介"
uv run timecampus-agent ask "检索主楼资料并给出维护计划"
uv run timecampus-agent ask --agent guide "主楼到图书馆怎么走？"
uv run timecampus-agent mcp-tools
uv run timecampus-agent eval run --suite all --mode fixture --repetitions 2 --report-dir eval-reports
uv run timecampus-agent serve
```

游客路线辅助：

```powershell
uv run timecampus-agent route "主楼,39.981,116.34;图书馆,39.982,116.341"
```

## CLI 命令

| 命令 | 用途 |
| --- | --- |
| `rag-search <query> [--limit N]` | 优先调用 MCP `timecampus_rag_search`，未配置 MCP token 时回退管理端 REST RAG |
| `draft <task> [--limit N]` | 调用 Backend `/admin/agent/draft` 生成 grounded 草案 |
| `ask [--agent auto\|operations\|guide] <prompt>` | 运行 LangGraph supervisor，自动或指定运营/导览智能体 |
| `route <name,lat,lng;...>` | 调用公开 `/map/walking-route` 生成游客步行路线摘要 |
| `mcp-tools` | 连接 Backend MCP Server 并列出工具 |
| `eval list` | 列出内置后台维护和游客导览评测用例 |
| `eval run` | 运行 Agent Eval Harness，输出 JSON/Markdown 报告并按门槛返回退出码 |
| `serve [--host HOST] [--port PORT]` | 启动供 Backend 代理调用的本地 Agent HTTP 服务 |

## Agent Evaluation

本模块内置一个轻量 Agent Eval Harness，面向 AI 产品测试场景设计，不绑定 LangSmith、DeepEval、Ragas 或 Promptfoo 等平台。它借鉴成熟评测框架的数据集、评估器、实验报告和 CI 门禁思路，但核心 runner、scorer 和 report 都在本仓库内实现。

核心对象：

- `EvalCase`：评测用例，包含 suite、target、input、expected、checks、tags 和 riskLevel。
- `AgentTrace`：运行轨迹，包含 output、toolCalls、带排名/分数的 retrievedDocs、routePlan、latencyMs 和 error。
- `EvalResult`：评分结果，包含 metrics、overall、passed、failureReasons 和 badCaseTags。

评测用例来自版本化 `evaluation/cases.jsonl`。默认 fixture 模式不访问网络，适合 CI：

```powershell
uv run timecampus-agent eval list
uv run timecampus-agent eval run --suite all --mode fixture --repetitions 2 --report-dir eval-reports --min-pass-rate 0.85 --min-overall 80 --min-consistency 0.80
```

Backend 启动后可以运行 live 模式：

```powershell
uv run timecampus-agent eval run --suite maintenance --mode live --repetitions 3
uv run timecampus-agent eval run --suite guide --mode live --repetitions 3
```

标记为 `rag-benchmark`、`target=retrieval` 的 16 条用例会直接调用 MCP
`timecampus_rag_search`，用 Recall@K、MRR、Hit@1 和 source diversity
隔离评估检索器；开放式 RAG 回答再由明确标记的 LLM Judge 评估
answer correctness 与 faithfulness。

报告输出：

- `eval-reports/eval-report.json`：CI 和机器解析。
- `eval-reports/eval-report.md`：面试展示、Bad Case 复盘和人工评审。
- `eval-reports/runs/*.json`：最近 20 次运行，可用于版本对比。
- `eval-reports/bad-cases.jsonl`：append-only Bad Case 生命周期。

可选 LLM-as-judge：

```env
TIMECAMPUS_EVAL_LLM_ENABLED=true
TIMECAMPUS_CHAT_API_KEY=<key>
```

LLM-as-judge 不在 Fixture/CI 中运行；对显式标记的 Live 用例，其分数会重新计算
overall 并参与质量门禁，缺失或解析失败会记录为失败。

## 架构

| 文件 | 职责 |
| --- | --- |
| `timecampus_agent.config` | 从 `.env` 加载 Backend、管理员、Chat Model、MCP 配置 |
| `timecampus_agent.backend` | Backend REST API typed client |
| `timecampus_agent.tools` | 运营/导览 LangChain tool wrappers |
| `timecampus_agent.mcp_client` | Backend MCP 工具加载 |
| `timecampus_agent.agent` | LangGraph supervisor 和运营/导览工具调用子图 |
| `timecampus_agent.evaluation` | Agent Eval Harness、用例、评分器、报告和 CLI |
| `timecampus_agent.service` | FastAPI 内部接口、MCP Adapter、HITL 审批和 Eval HTTP API |
| `timecampus_agent.memory` | JSONL 会话持久化、原子写入和 `MEMORY.md` 长期运营约束 |
| `timecampus_agent.cli` | 本地操作员 CLI |

`rag-search` 在生产配置 MCP token 后优先走 MCP RAG；`draft` 和 LangChain tools 仍保留 REST client，便于管理端草案、冒烟和测试。

## 会话与记忆

- `TIMECAMPUS_AGENT_MEMORY_DIR/sessions/*.jsonl` 保存会话元数据与用户/助手消息，服务重启后仍可选择原 session 续聊。
- `TIMECAMPUS_AGENT_MEMORY_DIR/MEMORY.md` 是可人工维护的长期运营约束，启动时注入运营执行器提示词，最大读取 8000 字符。
- 每次调用只回放最近 `TIMECAMPUS_AGENT_SESSION_HISTORY_LIMIT` 条消息，默认 40。
- JSONL 使用临时文件、`fsync` 和原子替换写入；会话 ID 只接受 UUID。
- LangGraph HITL checkpoint 仍使用 `InMemorySaver`，因此服务重启后历史会话可续聊，但未审批线程失效。
- Portal 通过 Backend SSE 代理接收 `status`、`preflight`、`delta`、`result`、`done` 事件。

## 安全边界

Agent 运行应遵守 Backend MCP Server 的相同规则：

- 维护建议前先检索 grounded context。
- 写入前读取当前记录。
- 文案编辑优先使用 copy-only 能力。
- Portal 运营页的写操作使用 LangGraph HITL 暂停，管理员只能批准或拒绝原始参数。
- `timecampus_delete_poi`、`timecampus_delete_media` 不会加载进运营执行器。
- 版权不明、年份/地点/来源不明确的任务必须停止或人工复核。
- 生产 MCP 必须使用 `TIMECAMPUS_MCP_TOKEN`。
- 内部 HTTP 服务必须使用 `TIMECAMPUS_AGENT_API_TOKEN`，浏览器不直接访问该服务。

## 测试

```powershell
uv run pytest
uv run ruff check .
uv run timecampus-agent eval run --suite all --mode fixture --repetitions 2 --report-dir eval-reports --min-pass-rate 0.85 --min-overall 80 --min-consistency 0.80
```

根仓库 API 冒烟：

```powershell
cd ..\TimeCampus
node tools\agent-smoke.mjs --dry-run
```

## 相关文档

- 项目功能规格：[../docs/functional-spec.md](../docs/functional-spec.md)
- 项目技术规格：[../docs/technical-spec.md](../docs/technical-spec.md)
- LangGraph Agent 状态图：[docs/langgraph-agent.md](docs/langgraph-agent.md)
- Agent Stack 联调：[../docs/agent-stack.md](../docs/agent-stack.md)
- Agent Evaluation：[../docs/agent-evaluation.md](../docs/agent-evaluation.md)
- Backend MCP Server：[../TimeCampus-Backend/docs/mcp-server.md](../TimeCampus-Backend/docs/mcp-server.md)
- 文档维护指南：[../docs/documentation-maintenance.md](../docs/documentation-maintenance.md)
