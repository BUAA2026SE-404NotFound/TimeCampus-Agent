from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, wrap_model_call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from timecampus_agent.agent import OPERATIONS_PROMPT
from timecampus_agent.config import Settings

BLOCKED_TOOLS = {"timecampus_delete_poi", "timecampus_delete_media"}
GUIDE_ONLY_TOOLS = {"timecampus_public_poi_search", "timecampus_walking_route"}
READ_ONLY_TOOLS = {
    "timecampus_search_pois",
    "timecampus_get_poi",
    "timecampus_list_media",
    "timecampus_get_media",
    "timecampus_rag_search",
    "timecampus_rag_context_pack",
    "timecampus_rag_corpus_summary",
    "timecampus_public_poi_search",
    "timecampus_walking_route",
}

EXECUTION_PROMPT = (
    OPERATIONS_PROMPT
    + """
Use MCP read tools before any write. Never call deletion tools. A write may only
run after an administrator approves the exact tool name and arguments. Keep the
final response concise and list the records changed. For every new maintenance
task, call timecampus_rag_search before other record tools and cite returned
timecampus:// URIs. Every answer based on RAG must end with a "Sources:" line
that copies at least one exact timecampus:// URI from the tool result. Do not
use visitor-guide POI or route tools. If RAG returns no hits, explicitly state
that no reliable source was found and that requested facts cannot be confirmed
or invented; do not draft those facts.
"""
)
URI_PATTERN = re.compile(r"timecampus://[^\s`\"',]+")


@wrap_model_call
async def enforce_rag_first(request: Any, handler: Any) -> Any:
    last_user = max(
        (
            index
            for index, message in enumerate(request.messages)
            if isinstance(message, HumanMessage)
        ),
        default=-1,
    )
    current_turn = request.messages[last_user + 1 :]
    has_rag_result = any(
        isinstance(message, ToolMessage) and message.name == "timecampus_rag_search"
        for message in current_turn
    )
    if last_user >= 0 and not has_rag_result:
        request = request.override(
            tool_choice={
                "type": "function",
                "function": {"name": "timecampus_rag_search"},
            }
        )

    response = await handler(request)
    if not response.result:
        return response
    final = response.result[-1]
    if not isinstance(final, AIMessage) or final.tool_calls:
        return response
    content = final.content if isinstance(final.content, str) else ""
    if "Sources:" in content:
        return response
    uris = list(
        dict.fromkeys(
            uri
            for message in current_turn
            if isinstance(message, ToolMessage)
            for uri in URI_PATTERN.findall(str(message.content))
        )
    )
    if uris:
        response.result[-1] = final.model_copy(
            update={"content": content.rstrip() + "\n\nSources: " + ", ".join(uris[:5])}
        )
    return response


def tool_policy(tools: list[Any]) -> tuple[list[Any], dict[str, Any]]:
    allowed = [tool for tool in tools if tool.name not in BLOCKED_TOOLS]
    interrupt_on = {
        tool.name: (
            False
            if tool.name in READ_ONLY_TOOLS
            else {"allowed_decisions": ["approve", "reject"]}
        )
        for tool in allowed
    }
    return allowed, interrupt_on


async def build_operations_mcp_agent(
    settings: Settings,
    *,
    memory_context: str = "",
    extra_middleware: list[Any] | None = None,
) -> tuple[Any, MultiServerMCPClient]:
    if not settings.chat_api_key:
        raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is not configured.")
    headers = (
        {"X-TimeCampus-MCP-Token": settings.mcp_token}
        if settings.mcp_token
        else {}
    )
    client = MultiServerMCPClient(
        {
            "timecampus": {
                "transport": "http",
                "url": settings.mcp_url,
                "headers": headers,
            }
        }
    )
    operations_tools = [
        tool for tool in await client.get_tools() if tool.name not in GUIDE_ONLY_TOOLS
    ]
    tools, interrupt_on = tool_policy(operations_tools)
    if not tools:
        raise RuntimeError("Backend MCP returned no tools.")
    prompt = EXECUTION_PROMPT
    if memory_context:
        prompt += f"\n\n# Persistent operator memory\n{memory_context}"
    model = ChatDeepSeek(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
    )
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        middleware=[
            enforce_rag_first,
            HumanInTheLoopMiddleware(
                interrupt_on=interrupt_on,
                description_prefix="TimeCampus write requires administrator approval",
            ),
            *(extra_middleware or []),
        ],
        checkpointer=InMemorySaver(),
        name="operations_executor",
    )
    return agent, client
