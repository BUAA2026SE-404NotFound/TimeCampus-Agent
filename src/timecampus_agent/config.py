from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    admin_username: str | None
    admin_password: str | None
    admin_token: str | None
    chat_base_url: str
    chat_api_key: str | None
    chat_model: str
    chat_temperature: float
    mcp_url: str
    mcp_token: str | None
    eval_llm_enabled: bool


def load_settings() -> Settings:
    load_dotenv(dotenv_path=".env")
    return Settings(
        api_base_url=_env("TIMECAMPUS_API_BASE_URL", "http://127.0.0.1:8080/api/v1"),
        admin_username=_optional_env("TIMECAMPUS_ADMIN_USERNAME"),
        admin_password=_optional_env("TIMECAMPUS_ADMIN_PASSWORD"),
        admin_token=_optional_env("TIMECAMPUS_ADMIN_TOKEN"),
        chat_base_url=_env("TIMECAMPUS_CHAT_BASE_URL", "https://api.deepseek.com/v1"),
        chat_api_key=_optional_env("TIMECAMPUS_CHAT_API_KEY"),
        chat_model=_env("TIMECAMPUS_CHAT_MODEL", "deepseek-chat"),
        chat_temperature=float(_env("TIMECAMPUS_CHAT_TEMPERATURE", "0.2")),
        mcp_url=_env("TIMECAMPUS_MCP_URL", "http://127.0.0.1:8080/mcp"),
        mcp_token=_optional_env("TIMECAMPUS_MCP_TOKEN"),
        eval_llm_enabled=_bool_env("TIMECAMPUS_EVAL_LLM_ENABLED", False),
    )


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
