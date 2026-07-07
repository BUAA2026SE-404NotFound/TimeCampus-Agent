from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal
from uuid import uuid4

from timecampus_agent.backend import TimeCampusBackendClient
from timecampus_agent.config import Settings, load_settings
from timecampus_agent.llm import ChatClient
from timecampus_agent.tools import ToolSpec, build_guide_tools, build_operations_tools

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
MAX_AGENT_STEPS = 8


class PythonAgentExecutor:
    def __init__(
        self,
        model: ChatClient,
        tools: list[ToolSpec],
        system_prompt: str,
        *,
        name: str,
    ) -> None:
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.name = name

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self.ainvoke(payload))

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(payload.get("messages", []))
        new_messages = await self.run_messages(messages)
        return {"messages": [*messages, *new_messages], "active_agent": self.name}

    async def run_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        thread_messages: list[dict[str, Any]] = []
        for _ in range(MAX_AGENT_STEPS):
            response = await self.model.complete(
                [{"role": "system", "content": self.system_prompt}, *messages, *thread_messages],
                [tool.openai_schema() for tool in self.tools.values()],
            )
            thread_messages.append(response)
            calls = _tool_calls(response)
            if not calls:
                return thread_messages
            for call in calls:
                thread_messages.append(await self._tool_message(call))
        thread_messages.append(
            {
                "role": "assistant",
                "content": "工具调用轮次过多，请缩小任务范围后重试。",
            }
        )
        return thread_messages

    async def _tool_message(self, call: dict[str, Any]) -> dict[str, Any]:
        name = _tool_name(call)
        arguments = _tool_arguments(call)
        tool = self.tools.get(name)
        if tool is None:
            content = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
        else:
            try:
                content = _content_text(await tool.ainvoke(arguments))
            except Exception as exception:
                content = json.dumps({"error": str(exception)}, ensure_ascii=False)
        return {
            "role": "tool",
            "tool_call_id": str(call.get("id") or uuid4()),
            "name": name,
            "content": content,
        }


class TimeCampusAgentExecutor:
    def __init__(
        self,
        operations_agent: PythonAgentExecutor,
        guide_agent: PythonAgentExecutor,
        default_agent: AgentName = "auto",
    ) -> None:
        self.operations_agent = operations_agent
        self.guide_agent = guide_agent
        self.default_agent = default_agent

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self.ainvoke(payload))

    async def ainvoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = _normalize_messages(payload.get("messages", []))
        active_agent, reason = route_agent(_last_message_text(messages), self.default_agent)
        agent = self.guide_agent if active_agent == "guide" else self.operations_agent
        result = await agent.ainvoke({"messages": messages})
        result["active_agent"] = active_agent
        result["route_reason"] = reason
        return result


def create_agent_executor(
    settings: Settings | None = None,
    default_agent: AgentName = "auto",
) -> TimeCampusAgentExecutor:
    settings = settings or load_settings()
    if not settings.chat_api_key:
        raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is required to run the Python agent.")

    client = TimeCampusBackendClient(settings.api_base_url, admin_token=settings.admin_token)
    if not client.admin_token and settings.admin_username and settings.admin_password:
        client.login(settings.admin_username, settings.admin_password)

    model = ChatClient(settings)
    return create_timecampus_agent(model, client, default_agent=default_agent)


def create_timecampus_agent(
    model: ChatClient,
    client: TimeCampusBackendClient,
    default_agent: AgentName = "auto",
) -> TimeCampusAgentExecutor:
    return TimeCampusAgentExecutor(
        PythonAgentExecutor(
            model,
            build_operations_tools(client),
            OPERATIONS_PROMPT,
            name="operations",
        ),
        PythonAgentExecutor(
            model,
            build_guide_tools(client),
            GUIDE_PROMPT,
            name="guide",
        ),
        default_agent=default_agent,
    )


def route_agent(prompt: str, default_agent: AgentName = "auto") -> tuple[str, str]:
    if default_agent in {"operations", "guide"}:
        return default_agent, "forced by caller"
    text = prompt.casefold()
    if POINT_PATTERN.search(prompt):
        return "guide", "coordinate route request"
    if any(marker in text for marker in GUIDE_MARKERS):
        return "guide", "visitor guide intent"
    return "operations", "default operations intent"


def _normalize_messages(messages: object) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role") or "user")
            content = _content_text(message.get("content", ""))
            item = {"role": role, "content": content}
            if message.get("tool_calls"):
                item["tool_calls"] = message["tool_calls"]
            if message.get("tool_call_id"):
                item["tool_call_id"] = str(message["tool_call_id"])
            if message.get("name"):
                item["name"] = str(message["name"])
            normalized.append(item)
    return normalized


def _last_message_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if content:
            return _content_text(content)
    return ""


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls")
    return [call for call in calls if isinstance(call, dict)] if isinstance(calls, list) else []


def _tool_name(call: dict[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(call.get("name") or "")


def _tool_arguments(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    raw = function.get("arguments") if isinstance(function, dict) else call.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)
