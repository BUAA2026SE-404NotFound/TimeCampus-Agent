from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, wrap_model_call
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
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
BULK_DRAFT_THRESHOLD = 4_000
MAX_WRITE_CALLS_PER_TURN = 8
BULK_DRAFT_INSTRUCTION = """
The current user message is bulk source material. This turn is draft-only:
- use only read tools and do not request any write;
- remove empty cross-references, duplicate entries, and headings without content;
- map usable facts to existing POIs and separate accepted candidates, excluded
  items, unresolved mappings, and batches of at most 8 POIs;
- exclude unverifiable claims, copyright-unclear material, personal allegations,
  sexual harassment, crime, and private-person anecdotes from public fun facts.
The administrator must explicitly select a later batch before writes are proposed.
"""

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
or invented; do not draft those facts. Do not narrate tool planning or expose
chain-of-thought. When requesting tools, emit tool calls without user-facing
prose and reserve Markdown content for the final response.
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
    current_user_text = (
        _message_text(request.messages[last_user]) if last_user >= 0 else ""
    )
    if len(current_user_text) > BULK_DRAFT_THRESHOLD:
        system_text = request.system_message.text if request.system_message else ""
        request = request.override(
            system_message=SystemMessage(
                content=system_text + "\n" + BULK_DRAFT_INSTRUCTION
            ),
            tools=[
                tool
                for tool in request.tools
                if getattr(tool, "name", "") in READ_ONLY_TOOLS
            ],
        )
    current_turn = request.messages[last_user + 1 :]
    has_rag_result = any(
        isinstance(message, ToolMessage) and message.name == "timecampus_rag_search"
        for message in current_turn
    )
    if last_user >= 0 and not has_rag_result:
        response = await handler(request)
        if any(
            call.get("name") == "timecampus_rag_search"
            for message in response.result
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ):
            return response
        system_text = request.system_message.text if request.system_message else ""
        retry_request = request.override(
            system_message=SystemMessage(
                content=system_text
                + "\nYou must call timecampus_rag_search before answering this turn."
            )
        )
        return await handler(retry_request)

    response = await handler(request)
    if not response.result:
        return response
    write_calls = [
        call
        for message in response.result
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call.get("name") not in READ_ONLY_TOOLS
    ]
    if len(write_calls) > MAX_WRITE_CALLS_PER_TURN:
        response.result[:] = [
            AIMessage(
                content=(
                    "本轮包含超过 8 个写操作。请明确选择最多 8 个 POI，"
                    "再分批生成待审批修改。"
                )
            )
        ]
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


def _message_text(message: BaseMessage) -> str:
    return message.content if isinstance(message.content, str) else ""


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
    checkpointer: Any | None = None,
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
        checkpointer=checkpointer or InMemorySaver(),
        name="operations_executor",
    )
    return agent, client
