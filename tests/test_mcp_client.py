from timecampus_agent.config import Settings
from timecampus_agent.mcp_client import build_mcp_client, _try_parse_streamable_payload


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
