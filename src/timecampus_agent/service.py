from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from timecampus_agent.agent import OPERATIONS_PROMPT
from timecampus_agent.config import Settings, load_settings
from timecampus_agent.evaluation.cases import case_catalog
from timecampus_agent.evaluation.models import EvalMode
from timecampus_agent.evaluation.reports import write_eval_report
from timecampus_agent.evaluation.runner import EvalRunner
from timecampus_agent.memory import SessionStore

BLOCKED_TOOLS = {"timecampus_delete_poi", "timecampus_delete_media"}
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
final response concise and list the records changed.
"""
)


class OperationRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, alias="sessionId")


class SessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=60)


class SessionMessageRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class OperationDecision(BaseModel):
    type: Literal["approve", "reject"]
    message: str | None = Field(default=None, max_length=1000)


class OperationDecisionRequest(BaseModel):
    decisions: list[OperationDecision] = Field(min_length=1, max_length=20)


class EvalRunRequest(BaseModel):
    suite: Literal["all", "maintenance", "guide"] = "all"
    mode: EvalMode = "fixture"
    min_pass_rate: float = Field(default=0.85, ge=0, le=1)
    min_overall: float = Field(default=80, ge=0, le=100)


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


class AgentRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sessions = SessionStore(
            Path(settings.memory_dir),
            history_limit=settings.session_history_limit,
        )
        self._agent: Any | None = None
        self._client: MultiServerMCPClient | None = None
        self._lock = asyncio.Lock()
        self._active_threads: set[str] = set()
        self._thread_sessions: dict[str, str] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    def list_sessions(self) -> list[dict[str, Any]]:
        return self.sessions.list()

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        return self.sessions.create(title)

    def get_session(self, session_id: str) -> dict[str, Any]:
        try:
            session = self.sessions.get(session_id)
        except ValueError as exception:
            raise HTTPException(status_code=404, detail="Agent session not found.") from exception
        if not session:
            raise HTTPException(status_code=404, detail="Agent session not found.")
        return session

    def record_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        self.get_session(session_id)
        return self.sessions.append(session_id, role, content)

    async def start(self, task: str, session_id: str | None = None) -> dict[str, Any]:
        session_id = session_id or self.create_session(task)["id"]
        async with self._session_lock(session_id):
            messages = self._append_user_and_history(session_id, task)
            base_message_count = len(messages)
            agent = await self._get_agent()
            thread_id = str(uuid4())
            self._active_threads.add(thread_id)
            self._thread_sessions[thread_id] = session_id
            result = await agent.ainvoke(
                {"messages": messages},
                config={"configurable": {"thread_id": thread_id}},
            )
            execution = self._serialize(
                thread_id,
                result,
                result.get("messages", [])[base_message_count:],
            )
            self._persist_assistant(session_id, execution["output"])
            execution["sessionId"] = session_id
            return execution

    async def stream_start(
        self,
        session_id: str,
        task: str,
    ) -> AsyncIterator[tuple[str, Any]]:
        async with self._session_lock(session_id):
            messages = self._append_user_and_history(session_id, task)
            agent = await self._get_agent()
            thread_id = str(uuid4())
            self._active_threads.add(thread_id)
            self._thread_sessions[thread_id] = session_id
            yield "session", self.sessions.summary(session_id)
            async for event in self._stream_agent(
                agent,
                {"messages": messages},
                thread_id,
                session_id,
                len(messages),
            ):
                yield event

    async def stream_resume(
        self,
        thread_id: str,
        decisions: list[OperationDecision],
    ) -> AsyncIterator[tuple[str, Any]]:
        session_id = self._thread_sessions.get(thread_id)
        if thread_id not in self._active_threads or not session_id:
            raise HTTPException(status_code=409, detail="Agent run expired; start it again.")
        async with self._session_lock(session_id):
            agent = await self._get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = await agent.aget_state(config)
            base_message_count = len(snapshot.values.get("messages", []))
            command = Command(
                resume={
                    "decisions": [
                        decision.model_dump(exclude_none=True) for decision in decisions
                    ]
                }
            )
            async for event in self._stream_agent(
                agent,
                command,
                thread_id,
                session_id,
                base_message_count,
            ):
                yield event

    async def _stream_agent(
        self,
        agent: Any,
        input_value: Any,
        thread_id: str,
        session_id: str,
        base_message_count: int,
    ) -> AsyncIterator[tuple[str, Any]]:
        config = {"configurable": {"thread_id": thread_id}}
        streamed: list[str] = []
        async for message, _metadata in agent.astream(
            input_value,
            config=config,
            stream_mode="messages",
        ):
            if isinstance(message, AIMessageChunk):
                delta = _content_text(message.content)
                if delta:
                    streamed.append(delta)
                    yield "delta", {"content": delta}
        snapshot = await agent.aget_state(config)
        result = dict(snapshot.values)
        result["__interrupt__"] = snapshot.interrupts
        new_messages = result.get("messages", [])[base_message_count:]
        output = "".join(streamed).strip() or _last_ai_text(new_messages)
        execution = self._serialize(thread_id, result, new_messages, output=output)
        self._persist_assistant(session_id, execution["output"])
        execution["sessionId"] = session_id
        yield "result", execution
        yield "done", {"status": execution["status"]}

    async def resume(
        self,
        thread_id: str,
        decisions: list[OperationDecision],
    ) -> dict[str, Any]:
        session_id = self._thread_sessions.get(thread_id)
        if thread_id not in self._active_threads or not session_id:
            raise HTTPException(status_code=409, detail="Agent run expired; start it again.")
        async with self._session_lock(session_id):
            agent = await self._get_agent()
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = await agent.aget_state(config)
            base_message_count = len(snapshot.values.get("messages", []))
            result = await agent.ainvoke(
                Command(
                    resume={
                        "decisions": [
                            decision.model_dump(exclude_none=True) for decision in decisions
                        ]
                    }
                ),
                config=config,
            )
            execution = self._serialize(
                thread_id,
                result,
                result.get("messages", [])[base_message_count:],
            )
            self._persist_assistant(session_id, execution["output"])
            execution["sessionId"] = session_id
            return execution

    def _append_user_and_history(
        self,
        session_id: str,
        task: str,
    ) -> list[dict[str, str]]:
        self.get_session(session_id)
        self.sessions.append(session_id, "user", task)
        return self.sessions.prompt_messages(session_id)

    def _persist_assistant(self, session_id: str, output: str) -> None:
        if output.strip():
            self.sessions.append(session_id, "assistant", output)

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._session_locks.setdefault(session_id, asyncio.Lock())

    async def _get_agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        async with self._lock:
            if self._agent is not None:
                return self._agent
            if not self.settings.chat_api_key:
                raise HTTPException(
                    status_code=503,
                    detail="TIMECAMPUS_CHAT_API_KEY is not configured.",
                )
            headers = (
                {"X-TimeCampus-MCP-Token": self.settings.mcp_token}
                if self.settings.mcp_token
                else {}
            )
            self._client = MultiServerMCPClient(
                {
                    "timecampus": {
                        "transport": "http",
                        "url": self.settings.mcp_url,
                        "headers": headers,
                    }
                }
            )
            tools, interrupt_on = tool_policy(await self._client.get_tools())
            if not tools:
                raise HTTPException(status_code=503, detail="Backend MCP returned no tools.")
            model = ChatDeepSeek(
                api_key=self.settings.chat_api_key,
                base_url=self.settings.chat_base_url,
                model=self.settings.chat_model,
                temperature=self.settings.chat_temperature,
            )
            memory = self.sessions.memory_context()
            prompt = EXECUTION_PROMPT
            if memory:
                prompt += f"\n\n# Persistent operator memory\n{memory}"
            self._agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=prompt,
                middleware=[
                    HumanInTheLoopMiddleware(
                        interrupt_on=interrupt_on,
                        description_prefix="TimeCampus write requires administrator approval",
                    )
                ],
                checkpointer=InMemorySaver(),
                name="operations_executor",
            )
            return self._agent

    def _serialize(
        self,
        thread_id: str,
        result: dict[str, Any],
        messages: list[BaseMessage],
        *,
        output: str | None = None,
    ) -> dict[str, Any]:
        pending = _pending_actions(result)
        payloads = [_message_payload(message) for message in messages]
        if pending:
            status = "approval_required"
        else:
            status = "completed"
            self._active_threads.discard(thread_id)
            self._thread_sessions.pop(thread_id, None)
        return {
            "threadId": thread_id,
            "status": status,
            "pendingActions": pending,
            "toolEvents": [message for message in payloads if message["type"] == "tool"],
            "output": output if output is not None else _last_ai_text(messages),
        }

def create_app(
    settings: Settings | None = None,
    runtime: AgentRuntime | Any | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    runtime = runtime or AgentRuntime(settings)
    app = FastAPI(title="TimeCampus Agent", version="0.3.0-beta")

    def authorize(
        token: str | None = Header(default=None, alias="X-TimeCampus-Agent-Token"),
    ) -> None:
        if not settings.agent_api_token:
            raise HTTPException(status_code=503, detail="Agent API token is not configured.")
        if token != settings.agent_api_token:
            raise HTTPException(status_code=401, detail="Invalid Agent API token.")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/internal/v1/operations/runs", dependencies=[Depends(authorize)])
    async def start_operation(request: OperationRunRequest) -> dict[str, Any]:
        return await runtime.start(request.task.strip(), request.session_id)

    @app.get("/internal/v1/operations/sessions", dependencies=[Depends(authorize)])
    async def list_sessions() -> dict[str, Any]:
        return {"sessions": runtime.list_sessions()}

    @app.post("/internal/v1/operations/sessions", dependencies=[Depends(authorize)])
    async def create_session(request: SessionCreateRequest) -> dict[str, Any]:
        return runtime.create_session(request.title)

    @app.get(
        "/internal/v1/operations/sessions/{session_id}",
        dependencies=[Depends(authorize)],
    )
    async def get_session(session_id: str) -> dict[str, Any]:
        return runtime.get_session(session_id)

    @app.post(
        "/internal/v1/operations/sessions/{session_id}/messages",
        dependencies=[Depends(authorize)],
    )
    async def record_session_message(
        session_id: str,
        request: SessionMessageRequest,
    ) -> dict[str, Any]:
        return runtime.record_message(session_id, request.role, request.content)

    @app.post(
        "/internal/v1/operations/sessions/{session_id}/messages/stream",
        dependencies=[Depends(authorize)],
    )
    async def stream_operation(
        session_id: str,
        request: OperationRunRequest,
    ) -> StreamingResponse:
        return _stream_response(runtime.stream_start(session_id, request.task.strip()))

    @app.post(
        "/internal/v1/operations/runs/{thread_id}/decisions",
        dependencies=[Depends(authorize)],
    )
    async def resume_operation(
        thread_id: str,
        request: OperationDecisionRequest,
    ) -> dict[str, Any]:
        return await runtime.resume(thread_id, request.decisions)

    @app.post(
        "/internal/v1/operations/runs/{thread_id}/decisions/stream",
        dependencies=[Depends(authorize)],
    )
    async def stream_resume_operation(
        thread_id: str,
        request: OperationDecisionRequest,
    ) -> StreamingResponse:
        return _stream_response(runtime.stream_resume(thread_id, request.decisions))

    @app.get("/internal/v1/evals/cases", dependencies=[Depends(authorize)])
    async def eval_cases(
        suite: Literal["all", "maintenance", "guide"] = "all",
    ) -> dict[str, Any]:
        return {"cases": case_catalog(suite)}

    @app.post("/internal/v1/evals/runs", dependencies=[Depends(authorize)])
    async def run_eval(request: EvalRunRequest) -> dict[str, Any]:
        summary = await asyncio.to_thread(
            EvalRunner(settings).run,
            request.suite,
            request.mode,
            request.min_pass_rate,
            request.min_overall,
        )
        await asyncio.to_thread(write_eval_report, summary, Path(settings.eval_report_dir))
        return jsonable_encoder(summary.model_dump(by_alias=True))

    return app


def _stream_response(events: AsyncIterator[tuple[str, Any]]) -> StreamingResponse:
    async def generate() -> AsyncIterator[str]:
        try:
            async for event, data in events:
                yield _sse(event, data)
        except HTTPException as exception:
            yield _sse("error", {"message": exception.detail, "status": exception.status_code})
        except Exception:
            yield _sse("error", {"message": "Agent stream failed.", "status": 500})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: Any) -> str:
    payload = json.dumps(jsonable_encoder(data), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _pending_actions(result: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in result.get("__interrupt__", ()):
        value = getattr(item, "value", item)
        if not isinstance(value, dict):
            continue
        requests = value.get("action_requests", [])
        configs = value.get("review_configs", [])
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                continue
            config = configs[index] if index < len(configs) else {}
            actions.append(
                {
                    "name": request.get("name", ""),
                    "arguments": request.get("arguments", request.get("args", {})),
                    "description": request.get("description", ""),
                    "allowedDecisions": config.get(
                        "allowed_decisions", ["approve", "reject"]
                    ),
                }
            )
    return actions


def _message_payload(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "tool" if isinstance(message, ToolMessage) else message.type,
        "content": _content_text(message.content),
    }
    if isinstance(message, ToolMessage):
        payload["name"] = message.name or ""
    return payload


def _last_ai_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return _content_text(message.content)
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    return str(content)
