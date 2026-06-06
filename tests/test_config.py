from timecampus_agent.config import load_settings


def test_load_settings_uses_defaults(monkeypatch) -> None:
    for name in [
        "TIMECAMPUS_API_BASE_URL",
        "TIMECAMPUS_CHAT_BASE_URL",
        "TIMECAMPUS_CHAT_MODEL",
        "TIMECAMPUS_MCP_URL",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.api_base_url == "http://127.0.0.1:8080/api/v1"
    assert settings.chat_base_url == "https://api.deepseek.com/v1"
    assert settings.chat_model == "deepseek-chat"
    assert settings.mcp_url == "http://127.0.0.1:8080/mcp"
