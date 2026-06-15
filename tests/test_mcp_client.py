import asyncio

from timecampus_agent.config import Settings
from timecampus_agent.mcp_client import McpStreamableHttpClient, build_mcp_client
from timecampus_agent.mcp_client import _try_parse_streamable_payload


def test_build_mcp_client_registers_timecampus_connection() -> None:
    settings = Settings(
        api_base_url="http://api.example.test",
        admin_username=None,
        admin_password=None,
        admin_token=None,
        chat_base_url="http://chat.example.test/v1",
        chat_api_key=None,
        chat_model="test-model",
        chat_temperature=0.2,
        mcp_url="http://mcp.example.test/mcp",
        mcp_token="token",
    )

    client = build_mcp_client(settings)

    assert client.url == "http://mcp.example.test/mcp"
    assert client.token == "token"


def test_parse_streamable_payload() -> None:
    payload = _try_parse_streamable_payload('event: message\ndata: {"jsonrpc":"2.0","id":1}\n\n')

    assert payload == {"jsonrpc": "2.0", "id": 1}


def test_call_tool_uses_mcp_tools_call_shape(monkeypatch) -> None:
    client = McpStreamableHttpClient("http://mcp.example.test/mcp", token="token")
    client.session_id = "session"
    calls = []

    async def fake_request(payload: dict, include_session: bool) -> dict:
        calls.append((payload, include_session))
        return {"result": {"structuredContent": {"ok": True}}}

    monkeypatch.setattr(client, "_request", fake_request)

    result = asyncio.run(client.call_tool("timecampus_rag_search", {"query": "主楼"}))

    assert result == {"result": {"structuredContent": {"ok": True}}}
    assert calls == [
        (
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "timecampus_rag_search",
                    "arguments": {"query": "主楼"},
                },
            },
            True,
        )
    ]
