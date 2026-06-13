# TimeCampus Agent

TimeCampus Agent 是“时光航迹”的独立 Python/LangChain 子模块，用于后台维护、RAG 检索、草案生成、MCP 工具发现和游客步行路线辅助。Agent 编排刻意放在 Portal 之外，前端保持纯 UI 客户端，Agent 通过 Backend REST API 和 `/mcp` 与系统交互。

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
```

`TIMECAMPUS_ADMIN_TOKEN` 优先于用户名密码；调用管理端 RAG 和草案接口时必须有管理员 token 或可登录凭据。

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
uv run timecampus-agent mcp-tools
```

游客路线辅助：

```powershell
uv run timecampus-agent route "主楼,39.981,116.34;图书馆,39.982,116.341"
```

## CLI 命令

| 命令 | 用途 |
| --- | --- |
| `rag-search <query> [--limit N]` | 调用 Backend `/admin/agent/rag/search` 检索 POI、影像、评论和 guideline |
| `draft <task> [--limit N]` | 调用 Backend `/admin/agent/draft` 生成 grounded 草案 |
| `ask <prompt>` | 组装 LangChain tool-calling agent 后回答 |
| `route <name,lat,lng;...>` | 调用公开 `/map/walking-route` 生成游客步行路线摘要 |
| `mcp-tools` | 连接 Backend MCP Server 并列出工具 |

## 架构

| 文件 | 职责 |
| --- | --- |
| `timecampus_agent.config` | 从 `.env` 加载 Backend、管理员、Chat Model、MCP 配置 |
| `timecampus_agent.backend` | Backend REST API typed client |
| `timecampus_agent.tools` | LangChain tool wrappers |
| `timecampus_agent.mcp_client` | Backend MCP 工具加载 |
| `timecampus_agent.agent` | LangChain tool-calling agent 组装 |
| `timecampus_agent.cli` | 本地操作员 CLI |

默认工具层优先走 Backend REST API，便于冒烟和测试；需要直接使用 MCP 时通过 `mcp-tools` 和 `timecampus_agent.mcp_client` 连接 Backend `/mcp`。

## 安全边界

Agent 运行应遵守 Backend MCP Server 的相同规则：

- 维护建议前先检索 grounded context。
- 写入前读取当前记录。
- 文案编辑优先使用 copy-only 能力。
- 删除、版权不明、年份/地点/来源不明确的任务必须人工确认。
- 生产 MCP 必须使用 `TIMECAMPUS_MCP_TOKEN`。

## 测试

```powershell
uv run pytest
uv run ruff check .
```

根仓库 API 冒烟：

```powershell
cd ..\TimeCampus
node tools\agent-smoke.mjs --dry-run
```

## 相关文档

- 项目功能规格：[../docs/functional-spec.md](../docs/functional-spec.md)
- 项目技术规格：[../docs/technical-spec.md](../docs/technical-spec.md)
- Agent Stack 联调：[../docs/agent-stack.md](../docs/agent-stack.md)
- Backend MCP Server：[../TimeCampus-Backend/docs/mcp-server.md](../TimeCampus-Backend/docs/mcp-server.md)
- 文档维护指南：[../docs/documentation-maintenance.md](../docs/documentation-maintenance.md)
