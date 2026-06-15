from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from timecampus_agent.config import Settings, load_settings


class McpClientError(RuntimeError):
    """Raised when the backend MCP endpoint returns an invalid response."""


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str | None = None


class McpStreamableHttpClient:
    def __init__(self, url: str, token: str | None = None, timeout: float = 15.0) -> None:
        self.url = url
        self.token = token
        self.timeout = timeout
        self.session_id: str | None = None

    async def initialize(self) -> dict[str, Any]:
        return await self._request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "timecampus-agent", "version": "0.3.0-beta"},
                },
            },
            include_session=False,
        )

    async def list_tools(self) -> list[McpToolInfo]:
        if not self.session_id:
            await self.initialize()
        payload = await self._request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            include_session=True,
        )
        tools = payload.get("result", {}).get("tools", [])
        if not isinstance(tools, list):
            raise McpClientError("MCP tools/list returned an invalid tools payload")
        return [
            McpToolInfo(
                name=str(tool.get("name", "")),
                description=tool.get("description"),
            )
            for tool in tools
            if isinstance(tool, dict) and tool.get("name")
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.session_id:
            await self.initialize()
        return await self._request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": name,
                    "arguments": arguments,
                },
            },
            include_session=True,
        )

    async def _request(self, payload: dict[str, Any], include_session: bool) -> dict[str, Any]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self.token:
            headers["X-TimeCampus-MCP-Token"] = self.token
        if include_session and self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.url, headers=headers, json=payload) as response:
                response.raise_for_status()
                session_id = response.headers.get("mcp-session-id")
                if session_id:
                    self.session_id = session_id
                chunks: list[str] = []
                async for chunk in response.aiter_text():
                    chunks.append(chunk)
                    parsed = _try_parse_streamable_payload("".join(chunks))
                    if parsed is not None:
                        return parsed
        raise McpClientError("MCP response ended without a JSON payload")


def build_mcp_client(settings: Settings | None = None) -> McpStreamableHttpClient:
    settings = settings or load_settings()
    return McpStreamableHttpClient(settings.mcp_url, token=settings.mcp_token)


async def list_timecampus_mcp_tool_names(settings: Settings | None = None) -> list[str]:
    client = build_mcp_client(settings)
    tools = await client.list_tools()
    return [tool.name for tool in tools]


async def call_timecampus_mcp_tool(
    name: str,
    arguments: dict[str, Any],
    settings: Settings | None = None,
) -> dict[str, Any]:
    client = build_mcp_client(settings)
    return await client.call_tool(name, arguments)


def _try_parse_streamable_payload(text: str) -> dict[str, Any] | None:
    data_lines = [
        line.removeprefix("data:").strip()
        for line in text.splitlines()
        if line.startswith("data:")
    ]
    candidate = "\n".join(data_lines).strip() if data_lines else text.strip()
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        raise McpClientError("MCP response JSON is not an object")
    return value
