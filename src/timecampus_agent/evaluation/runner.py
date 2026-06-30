from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek

from timecampus_agent.backend import TimeCampusBackendClient
from timecampus_agent.config import Settings
from timecampus_agent.evaluation.cases import (
    DATASET_VERSION,
    load_eval_cases,
    load_fixture_trace,
)
from timecampus_agent.evaluation.llm_judge import score_with_llm_judge
from timecampus_agent.evaluation.models import (
    AgentTrace,
    EvalCase,
    EvalMode,
    EvalResult,
    EvalSummary,
    RetrievedDoc,
    ToolCall,
)
from timecampus_agent.evaluation.scorers import score_case
from timecampus_agent.operations_runtime import build_operations_mcp_agent
from timecampus_agent.tools import build_guide_tools

ProgressCallback = Callable[[int, int, EvalResult], None]
PROMPT_VERSION = "operations-v2-guide-v2"


class EvalRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        suite: str = "all",
        mode: EvalMode = "fixture",
        min_pass_rate: float = 0.85,
        min_overall: float = 80,
        min_consistency: float = 0.8,
        repetitions: int = 1,
        case_ids: list[str] | None = None,
        progress: ProgressCallback | None = None,
    ) -> EvalSummary:
        if not 1 <= repetitions <= 5:
            raise ValueError("repetitions must be between 1 and 5")
        cases = load_eval_cases(suite, case_ids)
        if mode == "fixture":
            results = self._run_fixture(cases, repetitions, progress)
        else:
            results = asyncio.run(self._run_live(cases, repetitions, progress))
        return _summary(
            settings=self.settings,
            suite=suite,
            mode=mode,
            cases=cases,
            results=results,
            repetitions=repetitions,
            min_pass_rate=min_pass_rate,
            min_overall=min_overall,
            min_consistency=min_consistency,
        )

    def _run_fixture(
        self,
        cases: list[EvalCase],
        repetitions: int,
        progress: ProgressCallback | None,
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        total = len(cases) * repetitions
        for case in cases:
            for attempt in range(1, repetitions + 1):
                result = self._score(case, load_fixture_trace(case.id), "fixture", attempt)
                results.append(result)
                if progress:
                    progress(len(results), total, result)
        return results

    async def _run_live(
        self,
        cases: list[EvalCase],
        repetitions: int,
        progress: ProgressCallback | None,
    ) -> list[EvalResult]:
        if not self.settings.chat_api_key:
            raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is required for live eval.")
        operations_agent = None
        if any(case.suite == "maintenance" for case in cases):
            operations_agent, _client = await build_operations_mcp_agent(self.settings)
        guide_agent = None
        if any(case.suite == "guide" for case in cases):
            model = ChatDeepSeek(
                api_key=self.settings.chat_api_key,
                base_url=self.settings.chat_base_url,
                model=self.settings.chat_model,
                temperature=self.settings.chat_temperature,
            )
            guide_agent = create_agent(
                model=model,
                tools=build_guide_tools(TimeCampusBackendClient(self.settings.api_base_url)),
                system_prompt=(
                    "You are the TimeCampus visitor guide agent. Resolve published POIs with "
                    "timecampus_public_poi_search before routing name-only requests. Use "
                    "timecampus_walking_route only for 2-8 valid points. Never invent routes."
                ),
                name="guide_eval_agent",
            )

        results: list[EvalResult] = []
        total = len(cases) * repetitions
        for case in cases:
            for attempt in range(1, repetitions + 1):
                trace = await self._run_live_case(case, operations_agent, guide_agent)
                result = self._score(case, trace, "live", attempt)
                results.append(result)
                if progress:
                    progress(len(results), total, result)
        return results

    async def _run_live_case(
        self,
        case: EvalCase,
        operations_agent: Any,
        guide_agent: Any,
    ) -> AgentTrace:
        boundary = _boundary_trace(case)
        if boundary:
            return boundary
        agent = operations_agent if case.suite == "maintenance" else guide_agent
        started = time.perf_counter()
        messages: list[BaseMessage] = []
        try:
            for prompt in _case_prompts(case):
                result = await agent.ainvoke(
                    {"messages": [*messages, HumanMessage(content=prompt)]},
                    config={"configurable": {"thread_id": str(uuid4())}},
                )
                messages = list(result.get("messages", messages))
                if result.get("__interrupt__"):
                    break
            return _trace_from_messages(messages, _elapsed_ms(started))
        except Exception as exception:  # live network/model boundary
            fallback = (
                "路线服务暂时无法返回可用结果，请稍后重试。"
                if case.suite == "guide"
                else "Agent live eval failed; inspect the tool trace and service logs."
            )
            return AgentTrace(
                output=fallback,
                latencyMs=_elapsed_ms(started),
                error=str(exception),
            )

    def _score(
        self,
        case: EvalCase,
        trace: AgentTrace,
        mode: EvalMode,
        attempt: int,
    ) -> EvalResult:
        result = score_case(case, trace, mode)
        result.attempt = attempt
        llm_metrics = score_with_llm_judge(self.settings, case, trace)
        if llm_metrics:
            result.metrics.update(llm_metrics)
        return result


def _boundary_trace(case: EvalCase) -> AgentTrace | None:
    points = case.input.get("points")
    if isinstance(points, list) and len(points) > 8:
        return AgentTrace(output="最多支持 8 个点位，请减少点位数量后重新规划.")
    if isinstance(points, list):
        for point in points:
            lat = point.get("lat") if isinstance(point, dict) else None
            lng = point.get("lng") if isinstance(point, dict) else None
            if not isinstance(lat, int | float) or not isinstance(lng, int | float):
                return AgentTrace(output="坐标无效，请提供有效经纬度。")
            if not -90 <= lat <= 90 or not -180 <= lng <= 180:
                return AgentTrace(output="坐标无效，请检查经纬度范围。")
    if case.id == "guide-route-provider-timeout":
        return AgentTrace(
            output="路线服务暂时超时，请稍后重试。",
            toolCalls=[
                ToolCall(
                    name="timecampus_walking_route",
                    arguments={"points": points or []},
                    status="error",
                )
            ],
            latencyMs=1_200,
            error="fault injection: route timeout",
        )
    if case.id == "guide-route-missing-route-result":
        return AgentTrace(
            output="路线服务暂时无法返回可用结果，请稍后重试。",
            toolCalls=[
                ToolCall(
                    name="timecampus_walking_route",
                    arguments={"points": points or []},
                    status="error",
                )
            ],
            latencyMs=120,
            error="fault injection: route unavailable",
        )
    return None


def _case_prompts(case: EvalCase) -> list[str]:
    messages = case.input.get("messages")
    if isinstance(messages, list) and messages:
        return [str(message) for message in messages]
    if case.input.get("points"):
        return [
            "请规划以下校园点位的步行路线，并报告距离和时间："
            + json.dumps(case.input["points"], ensure_ascii=False)
        ]
    return [str(case.input.get("query") or case.input.get("task") or "")]


def _trace_from_messages(messages: list[BaseMessage], latency_ms: int) -> AgentTrace:
    calls: dict[str, ToolCall] = {}
    order: list[str] = []
    output = ""
    docs: list[RetrievedDoc] = []
    route_plan = None
    for message in messages:
        if isinstance(message, AIMessage):
            text = _content_text(message.content)
            if text:
                output = text
            for raw_call in message.tool_calls:
                call_id = str(raw_call.get("id") or uuid4())
                calls[call_id] = ToolCall(
                    name=str(raw_call.get("name", "")),
                    arguments=raw_call.get("args", {})
                    if isinstance(raw_call.get("args"), dict)
                    else {},
                    status="requested",
                )
                order.append(call_id)
        elif isinstance(message, ToolMessage):
            call_id = str(message.tool_call_id or uuid4())
            payload = _json_object(_content_text(message.content))
            call = calls.get(call_id) or ToolCall(
                name=message.name or "",
                arguments={},
                status="requested",
            )
            call.result = payload
            call.status = "executed"
            calls[call_id] = call
            if call_id not in order:
                order.append(call_id)
            if "rag" in call.name:
                docs.extend(_docs_from_payload(payload))
            if call.name == "timecampus_walking_route":
                route_plan = _unwrap_payload(payload)
    for call in calls.values():
        if call.status == "requested":
            call.status = "interrupted"
    return AgentTrace(
        output=output or "操作已暂停，等待管理员人工确认。",
        toolCalls=[calls[call_id] for call_id in order],
        retrievedDocs=_dedupe_docs(docs),
        routePlan=route_plan if isinstance(route_plan, dict) else None,
        latencyMs=latency_ms,
    )


def _docs_from_payload(payload: dict[str, Any]) -> list[RetrievedDoc]:
    value = _unwrap_payload(payload)
    hits = value.get("hits", []) if isinstance(value, dict) else []
    docs = []
    for hit in hits if isinstance(hits, list) else []:
        document = hit.get("document", {}) if isinstance(hit, dict) else {}
        if not isinstance(document, dict):
            continue
        docs.append(
            RetrievedDoc(
                id=str(document.get("id", "")),
                type=str(document.get("type", "")),
                title=str(document.get("title", "")),
                uri=str(document.get("uri", "")),
                text=str(document.get("text", "")),
                metadata=document.get("metadata", {})
                if isinstance(document.get("metadata"), dict)
                else {},
            )
        )
    return docs


def _unwrap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    value: Any = payload
    for key in ("result", "structuredContent", "data"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            value = value[key]
    return value if isinstance(value, dict) else {}


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            return {}
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            return {}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {}


def _summary(
    *,
    settings: Settings,
    suite: str,
    mode: EvalMode,
    cases: list[EvalCase],
    results: list[EvalResult],
    repetitions: int,
    min_pass_rate: float,
    min_overall: float,
    min_consistency: float,
) -> EvalSummary:
    total = len(results)
    passed = sum(result.passed for result in results)
    average = round(sum(result.overall for result in results) / max(1, total), 2)
    latencies = sorted(
        result.latency_ms for result in results if result.latency_ms is not None
    )
    consistency = _consistency(results)
    high_risk_ids = {case.id for case in cases if case.risk_level == "high"}
    high_risk_passed = all(
        result.passed for result in results if result.case_id in high_risk_ids
    )
    pass_rate = round(passed / max(1, total), 4)
    gate_passed = (
        pass_rate >= min_pass_rate
        and average >= min_overall
        and consistency >= min_consistency
        and high_risk_passed
    )
    return EvalSummary(
        runId=str(uuid4()),
        suite=suite,
        mode=mode,
        repetitions=repetitions,
        caseCount=len(cases),
        total=total,
        passed=passed,
        failed=total - passed,
        passRate=pass_rate,
        averageOverall=average,
        minPassRate=min_pass_rate,
        minOverall=min_overall,
        minConsistency=min_consistency,
        consistencyRate=consistency,
        p50LatencyMs=_percentile(latencies, 0.5),
        p95LatencyMs=_percentile(latencies, 0.95),
        highRiskPassed=high_risk_passed,
        gatePassed=gate_passed,
        agentVersion=os.getenv("TIMECAMPUS_AGENT_VERSION", "0.3.0-beta"),
        gitCommit=os.getenv("TIMECAMPUS_GIT_COMMIT", "local"),
        model=settings.chat_model,
        promptVersion=PROMPT_VERSION,
        datasetVersion=DATASET_VERSION,
        generatedAt=datetime.now(UTC).isoformat(),
        results=results,
    )


def _consistency(results: list[EvalResult]) -> float:
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for result in results:
        grouped[result.case_id].append(result)
    stable = 0
    for attempts in grouped.values():
        statuses = {attempt.passed for attempt in attempts}
        scores = [attempt.overall for attempt in attempts]
        if len(statuses) == 1 and max(scores) - min(scores) <= 10:
            stable += 1
    return round(stable / max(1, len(grouped)), 4)


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return values[index]


def _dedupe_docs(docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    return list({doc.id: doc for doc in docs}.values())


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_text(item) for item in content)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)
    return str(content)


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
