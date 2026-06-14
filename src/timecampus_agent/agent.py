from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from timecampus_agent.backend import TimeCampusBackendClient
from timecampus_agent.config import Settings, load_settings
from timecampus_agent.tools import build_backend_tools

SYSTEM_PROMPT = """You are the TimeCampus maintenance agent.

Use backend tools as the source of truth. Retrieve grounded context before
suggesting POI, media, copy, route, or review changes. Never invent dates,
locations, sources, copyright status, people, or backend IDs. For destructive
or uncertain actions, produce a review plan instead of executing writes.
"""


def create_agent_executor(settings: Settings | None = None) -> Any:
    settings = settings or load_settings()
    if not settings.chat_api_key:
        raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is required to run the LangChain agent.")

    client = TimeCampusBackendClient(settings.api_base_url, admin_token=settings.admin_token)
    if not client.admin_token and settings.admin_username and settings.admin_password:
        client.login(settings.admin_username, settings.admin_password)

    llm = ChatOpenAI(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
    )
    tools = build_backend_tools(client)
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
