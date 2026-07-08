from __future__ import annotations

import inspect
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from uuid import uuid4

from timecampus_agent.agent import OPERATIONS_PROMPT
from timecampus_agent.config import Settings
from timecampus_agent.llm import ChatClient
from timecampus_agent.mcp_client import McpStreamableHttpClient
from timecampus_agent.tools import ToolSpec

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
MAX_AGENT_STEPS = 8
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
ToolOverride = Callable[[str, dict[str, Any]], Any | Awaitable[Any | None] | None]


class PurePythonOperationsAgent:
    def __init__(
        self,
        model: ChatClient,
        tools: list[ToolSpec],
        prompt: str,
        *,
        tool_overrides: list[ToolOverride] | None = None,
    ) -> None:
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.prompt = prompt
        self.tool_overrides = tool_overrides or []

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        user_text = _last_user_text(messages)
        available = self._available_tools(user_text)
        turn_messages: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        await self._rag_first(user_text, turn_messages, tool_events)

        for _ in range(MAX_AGENT_STEPS):
            response = await self._complete(
                [{"role": "system", "content": self._prompt(user_text)}, *messages, *turn_messages],
                [tool.openai_schema() for tool in available.values()],
            )
            turn_messages.append(response)
            calls = _tool_calls(response)
            if not calls:
                output = _append_sources(_content_text(response.get("content")), turn_messages)
                turn_messages[-1] = {**response, "content": output}
                return _completed(thread_id, turn_messages, tool_events, output)

            write_calls = [call for call in calls if _is_write_call(call, available)]
            if len(write_calls) > MAX_WRITE_CALLS_PER_TURN:
                output = "本轮包含超过 8 个写操作。请明确选择最多 8 个 POI，再分批生成待审批修改。"
                turn_messages.append({"role": "assistant", "content": output})
                return _completed(thread_id, turn_messages, tool_events, output)
            if write_calls:
                pending = [_pending_action(call, available[_tool_name(call)]) for call in write_calls]
                output = _content_text(response.get("content")) or "等待管理员审批。"
                return {
                    "threadId": thread_id,
                    "status": "approval_required",
                    "messages": turn_messages,
                    "pendingActions": pending,
                    "toolEvents": tool_events,
                    "output": output,
                    "pendingState": {
                        "messages": [*messages, *turn_messages],
                        "pendingToolCalls": write_calls,
                    },
                }

            for call in calls:
                message, event = await self._execute_tool_call(call, available)
                turn_messages.append(message)
                if event:
                    tool_events.append(event)

        output = "工具调用轮次过多，请缩小任务范围后重试。"
        turn_messages.append({"role": "assistant", "content": output})
        return _completed(thread_id, turn_messages, tool_events, output)

    async def stream_run(
        self,
        messages: list[dict[str, Any]],
        *,
        thread_id: str,
    ) -> AsyncIterator[tuple[str, Any]]:
        user_text = _last_user_text(messages)
        available = self._available_tools(user_text)
        turn_messages: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        await self._rag_first(user_text, turn_messages, tool_events)

        for _ in range(MAX_AGENT_STEPS):
            prompt_messages = [{"role": "system", "content": self._prompt(user_text)}, *messages, *turn_messages]
            tools = [tool.openai_schema() for tool in available.values()]
            try:
                response = await self.model.complete(prompt_messages, tools)
            except Exception:
                async for event in self._stream_fallback(prompt_messages, turn_messages, tool_events, thread_id):
                    yield event
                return

            turn_messages.append(response)
            calls = _tool_calls(response)
            if not calls:
                output = _append_sources(_content_text(response.get("content")), turn_messages)
                turn_messages[-1] = {**response, "content": output}
                if output:
                    yield "delta", {"content": output}
                yield "result", _completed(thread_id, turn_messages, tool_events, output)
                return

            write_calls = [call for call in calls if _is_write_call(call, available)]
            if len(write_calls) > MAX_WRITE_CALLS_PER_TURN:
                output = "本轮包含超过 8 个写操作。请明确选择最多 8 个 POI，再分批生成待审批修改。"
                turn_messages.append({"role": "assistant", "content": output})
                yield "delta", {"content": output}
                yield "result", _completed(thread_id, turn_messages, tool_events, output)
                return
            if write_calls:
                yield "result", {
                    "threadId": thread_id,
                    "status": "approval_required",
                    "messages": turn_messages,
                    "pendingActions": [_pending_action(call, available[_tool_name(call)]) for call in write_calls],
                    "toolEvents": tool_events,
                    "output": _content_text(response.get("content")) or "等待管理员审批。",
                    "pendingState": {
                        "messages": [*messages, *turn_messages],
                        "pendingToolCalls": write_calls,
                    },
                }
                return

            for call in calls:
                message, event = await self._execute_tool_call(call, available)
                turn_messages.append(message)
                if event:
                    tool_events.append(event)

        output = "工具调用轮次过多，请缩小任务范围后重试。"
        turn_messages.append({"role": "assistant", "content": output})
        yield "delta", {"content": output}
        yield "result", _completed(thread_id, turn_messages, tool_events, output)

    async def resume(
        self,
        state: dict[str, Any],
        decisions: list[dict[str, Any]],
        *,
        thread_id: str,
    ) -> dict[str, Any]:
        messages = [item for item in state.get("messages", []) if isinstance(item, dict)]
        calls = [item for item in state.get("pendingToolCalls", []) if isinstance(item, dict)]
        tool_messages: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        for index, call in enumerate(calls):
            decision = decisions[min(index, len(decisions) - 1)] if decisions else {"type": "reject"}
            if decision.get("type") == "approve":
                message, event = await self._execute_tool_call(call, self.tools)
            else:
                message = {
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or uuid4()),
                    "name": _tool_name(call),
                    "content": json.dumps(
                        {"status": "rejected", "message": decision.get("message") or ""},
                        ensure_ascii=False,
                    ),
                }
                event = _tool_event(message, status="interrupted")
            tool_messages.append(message)
            if event:
                tool_events.append(event)

        try:
            response = await self.model.complete(
                [
                    {"role": "system", "content": self.prompt},
                    *messages,
                    *tool_messages,
                    {
                        "role": "user",
                        "content": "请基于审批结果生成简短中文总结，只列出已执行、已拒绝和后续建议。",
                    },
                ],
                None,
            )
            output = _content_text(response.get("content")) or _resume_fallback(tool_events)
            result_messages = [*tool_messages, {**response, "content": output}]
        except Exception:
            output = _resume_fallback(tool_events)
            result_messages = [*tool_messages, {"role": "assistant", "content": output}]

        return _completed(thread_id, result_messages, tool_events, output)

    def _available_tools(self, user_text: str) -> dict[str, ToolSpec]:
        tools = {
            name: tool
            for name, tool in self.tools.items()
            if name not in BLOCKED_TOOLS and name not in GUIDE_ONLY_TOOLS
        }
        if len(user_text) > BULK_DRAFT_THRESHOLD:
            tools = {name: tool for name, tool in tools.items() if name in READ_ONLY_TOOLS}
        return tools

    def _prompt(self, user_text: str) -> str:
        if len(user_text) > BULK_DRAFT_THRESHOLD:
            return self.prompt + "\n" + BULK_DRAFT_INSTRUCTION
        return self.prompt

    async def _rag_first(
        self,
        user_text: str,
        turn_messages: list[dict[str, Any]],
        tool_events: list[dict[str, Any]],
    ) -> None:
        tool = self.tools.get("timecampus_rag_search")
        if tool is None:
            return
        call = _tool_call(
            "timecampus_rag_search",
            {
                "query": user_text[:2_000] or "TimeCampus maintenance task",
                "limit": 6,
                "types": ["poi", "media", "comment", "guideline", "knowledge"],
                "includePending": True,
            },
        )
        turn_messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        message, event = await self._execute_tool_call(call, self.tools)
        turn_messages.append(message)
        if event:
            tool_events.append(event)

    async def _execute_tool_call(
        self,
        call: dict[str, Any],
        available: dict[str, ToolSpec],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        name = _tool_name(call)
        tool = available.get(name)
        if tool is None or name in BLOCKED_TOOLS:
            content = json.dumps({"error": f"Tool is not allowed: {name}"}, ensure_ascii=False)
            message = _tool_message(call, name, content)
            return message, _tool_event(message, status="error")
        try:
            content = _content_text(await self._call_tool(tool, _tool_arguments(call)))
            message = _tool_message(call, name, content)
            return message, _tool_event(message, status="executed")
        except Exception as exception:
            content = json.dumps({"error": str(exception)}, ensure_ascii=False)
            message = _tool_message(call, name, content)
            return message, _tool_event(message, status="error")

    async def _call_tool(self, tool: ToolSpec, arguments: dict[str, Any]) -> Any:
        for override in self.tool_overrides:
            value = override(tool.name, arguments)
            if inspect.isawaitable(value):
                value = await value
            if value is not None:
                return value
        return await tool.ainvoke(arguments)

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            return await self.model.complete(messages, tools)
        except Exception:
            if not tools:
                raise
            # ponytail: keep read-only answers alive if the provider rejects MCP tool schemas.
            return await self.model.complete(_plain_messages(messages), None)

    async def _stream_fallback(
        self,
        messages: list[dict[str, Any]],
        turn_messages: list[dict[str, Any]],
        tool_events: list[dict[str, Any]],
        thread_id: str,
    ) -> AsyncIterator[tuple[str, Any]]:
        output = ""
        plain_messages = _plain_messages(messages)
        try:
            async for chunk in self.model.stream_complete(plain_messages, None):
                output += chunk
                yield "delta", {"content": chunk}
        except Exception:
            if output:
                raise
            output = _content_text((await self.model.complete(plain_messages, None)).get("content"))
            if output:
                yield "delta", {"content": output}
        completed = _append_sources(output, turn_messages)
        suffix = completed.removeprefix(output)
        if suffix:
            yield "delta", {"content": suffix}
        turn_messages.append({"role": "assistant", "content": completed})
        yield "result", _completed(thread_id, turn_messages, tool_events, completed)


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


def _plain_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    plain: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = _content_text(message.get("content"))
        calls = _tool_calls(message)
        if calls:
            names = ", ".join(_tool_name(call) for call in calls)
            content = "\n".join(part for part in (content, f"Tool calls requested: {names}") if part)
        if role == "tool":
            name = str(message.get("name") or "tool")
            content = f"Tool result from {name}:\n{content}"
            role = "user"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        plain.append({"role": role, "content": content})
    return plain


async def build_operations_mcp_agent(
    settings: Settings,
    *,
    memory_context: str = "",
    tool_overrides: list[ToolOverride] | None = None,
) -> tuple[PurePythonOperationsAgent, McpStreamableHttpClient]:
    if not settings.chat_api_key:
        raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is not configured.")
    client = McpStreamableHttpClient(settings.mcp_url)
    tools = [
        _mcp_tool(client, info)
        for info in await client.list_tools()
        if info.name not in GUIDE_ONLY_TOOLS and info.name not in BLOCKED_TOOLS
    ]
    if not tools:
        raise RuntimeError("Backend MCP returned no tools.")
    prompt = EXECUTION_PROMPT
    if memory_context:
        prompt += f"\n\n# Persistent operator memory\n{memory_context}"
    return (
        PurePythonOperationsAgent(
            ChatClient(settings),
            tools,
            prompt,
            tool_overrides=tool_overrides,
        ),
        client,
    )


def _mcp_tool(client: McpStreamableHttpClient, info: Any) -> ToolSpec:
    async def call(arguments: dict[str, Any]) -> str:
        return json.dumps(_extract_mcp_tool_result(await client.call_tool(info.name, arguments)), ensure_ascii=False)

    return ToolSpec(
        name=info.name,
        description=info.description or "",
        parameters=info.input_schema or {"type": "object", "properties": {}},
        handler=call,
    )


def _extract_mcp_tool_result(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        return result
    structured = result.get("structuredContent")
    if structured is not None:
        return structured
    content = result.get("content")
    if isinstance(content, list):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        ]
        if len(texts) == 1:
            try:
                return json.loads(str(texts[0]))
            except json.JSONDecodeError:
                return {"text": texts[0]}
        if texts:
            return {"text": "\n".join(str(text) for text in texts)}
    return result


def _completed(
    thread_id: str,
    messages: list[dict[str, Any]],
    tool_events: list[dict[str, Any]],
    output: str,
) -> dict[str, Any]:
    return {
        "threadId": thread_id,
        "status": "completed",
        "messages": messages,
        "pendingActions": [],
        "toolEvents": tool_events,
        "output": output,
        "pendingState": None,
    }


def _pending_action(call: dict[str, Any], tool: ToolSpec) -> dict[str, Any]:
    return {
        "name": tool.name,
        "arguments": _tool_arguments(call),
        "description": tool.description,
        "allowedDecisions": ["approve", "reject"],
    }


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"call-{uuid4().hex}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _tool_message(call: dict[str, Any], name: str, content: str) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": str(call.get("id") or uuid4()),
        "name": name,
        "content": content,
    }


def _tool_event(message: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "type": "tool",
        "name": message.get("name", ""),
        "content": _content_text(message.get("content")),
        "status": status,
    }


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


def _is_write_call(call: dict[str, Any], available: dict[str, ToolSpec]) -> bool:
    name = _tool_name(call)
    return name in available and name not in READ_ONLY_TOOLS


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _content_text(message.get("content"))
    return ""


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _append_sources(content: str, messages: list[dict[str, Any]]) -> str:
    if "Sources:" in content:
        return content
    uris = list(
        dict.fromkeys(
            uri
            for message in messages
            if message.get("role") == "tool"
            for uri in URI_PATTERN.findall(_content_text(message.get("content")))
        )
    )
    return content.rstrip() + ("\n\nSources: " + ", ".join(uris[:5]) if uris else "")


def _resume_fallback(tool_events: list[dict[str, Any]]) -> str:
    approved = [event["name"] for event in tool_events if event.get("status") == "executed"]
    rejected = [event["name"] for event in tool_events if event.get("status") == "interrupted"]
    parts = []
    if approved:
        parts.append("已执行：" + "、".join(approved))
    if rejected:
        parts.append("已拒绝：" + "、".join(rejected))
    return "；".join(parts) or "审批已处理。"
