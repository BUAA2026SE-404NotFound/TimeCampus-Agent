from __future__ import annotations

import re
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, MessagesState, StateGraph

from timecampus_agent.backend import TimeCampusBackendClient
from timecampus_agent.config import Settings, load_settings
from timecampus_agent.tools import build_guide_tools, build_operations_tools

AgentName = Literal["auto", "operations", "guide"]

OPERATIONS_PROMPT = """You are the TimeCampus operations intelligence agent.

Use backend tools as the source of truth. Retrieve grounded context before
suggesting POI, media, copy, review, index, or maintenance changes. Prefer
timecampus_admin_draft for maintenance plans because it returns RAG context and
quality gates. Never invent dates, locations, sources, copyright status,
people, or backend IDs. For destructive or uncertain actions, produce a review
plan instead of executing writes.
"""

GUIDE_PROMPT = """You are the TimeCampus visitor guide agent.

Help visitors plan campus walks. Use timecampus_public_poi_search to resolve
published POIs when the user gives names, and use timecampus_walking_route only
when you have 2-8 valid GCJ-02 points. Reject invalid coordinates or too many
points with a short fix request. Keep answers practical for a campus visitor.
"""

GUIDE_MARKERS = (
    "route",
    "walking",
    "walk",
    "tour",
    "visitor",
    "guide",
    "路线",
    "怎么走",
    "步行",
    "导览",
    "游览",
    "游客",
    "参观",
)
POINT_PATTERN = re.compile(r"[^,;，；]+[,，]\s*-?\d+(?:\.\d+)?[,，]\s*-?\d+(?:\.\d+)?")


class TimeCampusAgentState(MessagesState, total=False):
    active_agent: str
    route_reason: str


def create_agent_executor(
    settings: Settings | None = None,
    default_agent: AgentName = "auto",
) -> Any:
    settings = settings or load_settings()
    if not settings.chat_api_key:
        raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is required to run the LangGraph agent.")

    client = TimeCampusBackendClient(settings.api_base_url, admin_token=settings.admin_token)
    if not client.admin_token and settings.admin_username and settings.admin_password:
        client.login(settings.admin_username, settings.admin_password)

    llm = ChatDeepSeek(
        api_key=settings.chat_api_key,
        base_url=settings.chat_base_url,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
    )
    return create_timecampus_graph(llm, client, default_agent=default_agent)


def create_timecampus_graph(
    llm: Any,
    client: TimeCampusBackendClient,
    default_agent: AgentName = "auto",
) -> Any:
    operations_agent = create_agent(
        model=llm,
        tools=build_operations_tools(client),
        system_prompt=OPERATIONS_PROMPT,
        name="operations_agent",
    )
    guide_agent = create_agent(
        model=llm,
        tools=build_guide_tools(client),
        system_prompt=GUIDE_PROMPT,
        name="guide_agent",
    )

    def supervisor(state: TimeCampusAgentState) -> dict[str, str]:
        prompt = _last_message_text(state.get("messages", []))
        active_agent, reason = route_agent(prompt, default_agent=default_agent)
        return {"active_agent": active_agent, "route_reason": reason}

    def operations_node(state: TimeCampusAgentState) -> dict[str, list[BaseMessage] | str]:
        result = operations_agent.invoke({"messages": state.get("messages", [])})
        return {"messages": [_last_graph_message(result)], "active_agent": "operations"}

    def guide_node(state: TimeCampusAgentState) -> dict[str, list[BaseMessage] | str]:
        result = guide_agent.invoke({"messages": state.get("messages", [])})
        return {"messages": [_last_graph_message(result)], "active_agent": "guide"}

    builder = StateGraph(TimeCampusAgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("operations_agent", operations_node)
    builder.add_node("guide_agent", guide_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("active_agent", "operations"),
        {
            "operations": "operations_agent",
            "guide": "guide_agent",
        },
    )
    builder.add_edge("operations_agent", END)
    builder.add_edge("guide_agent", END)
    return builder.compile()


def route_agent(prompt: str, default_agent: AgentName = "auto") -> tuple[str, str]:
    if default_agent in {"operations", "guide"}:
        return default_agent, "forced by caller"
    text = prompt.casefold()
    if POINT_PATTERN.search(prompt):
        return "guide", "coordinate route request"
    if any(marker in text for marker in GUIDE_MARKERS):
        return "guide", "visitor guide intent"
    return "operations", "default operations intent"


def _last_graph_message(result: object) -> BaseMessage:
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            message = messages[-1]
            if isinstance(message, BaseMessage):
                return message
            if isinstance(message, dict):
                return AIMessage(content=str(message.get("content", "")))
    return AIMessage(content=str(result))


def _last_message_text(messages: list[BaseMessage] | list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if content:
            return _content_to_text(content)
    return ""


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    return str(content)
