# TimeCampus Agent

Standalone LangChain agent workspace for TimeCampus maintenance and guide workflows.

This project intentionally keeps agent orchestration outside `TimeCampus-Portal`.
The frontend remains a pure UI client, while this package talks to the backend
admin/public APIs and can later load MCP tools directly.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Fill `.env` with the local backend URL, admin credentials/token, and an
OpenAI-compatible chat model key.

## Run

Start the backend MCP/API server first:

```powershell
cd ..\TimeCampus
.\tools\start-backend-mcp.ps1
```

Then run agent commands:

```powershell
cd TimeCampus-Agent
uv run timecampus-agent rag-search "main building copy"
uv run timecampus-agent draft "Improve the visitor-facing copy for the main building"
uv run timecampus-agent ask "Find context for editing the main building POI, then draft a safe update plan"
uv run timecampus-agent mcp-tools
```

Visitor route helper:

```powershell
uv run timecampus-agent route "Main Building,39.981,116.34;Library,39.982,116.341"
```

## Architecture

- `timecampus_agent.backend`: typed backend API client.
- `timecampus_agent.tools`: LangChain tool wrappers around backend capabilities.
- `timecampus_agent.mcp_client`: optional LangChain MCP tool loader for the backend MCP server.
- `timecampus_agent.agent`: LangChain tool-calling agent assembly.
- `timecampus_agent.cli`: local operator CLI.

The default tool layer uses backend REST APIs because they are stable and easy
to smoke-test. `mcp-tools` and `timecampus_agent.mcp_client` provide the MCP
bridge for agents that should call the backend MCP server directly.

## Quality Bar

Agent runs should follow the same safety rules as the backend MCP server:

- retrieve grounded context before any maintenance suggestion;
- read current records before write operations;
- use copy-only operations for copy edits where possible;
- require human confirmation for deletes, copyright uncertainty, and unclear
  year/location/source data.

## Test

```powershell
uv run pytest
```
