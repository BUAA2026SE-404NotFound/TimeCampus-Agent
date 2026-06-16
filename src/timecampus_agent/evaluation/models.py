from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["low", "medium", "high"]
EvalSuite = Literal["maintenance", "guide"]
EvalMode = Literal["fixture", "live"]


class ToolCall(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None


class RetrievedDoc(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: str
    title: str
    uri: str
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list, alias="toolCalls")
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list, alias="retrievedDocs")
    route_plan: dict[str, Any] | None = Field(default=None, alias="routePlan")
    latency_ms: int | None = Field(default=None, alias="latencyMs")
    error: str | None = None


class EvalCase(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    suite: EvalSuite
    target: str
    input: dict[str, Any]
    expected: dict[str, Any] = Field(default_factory=dict)
    checks: list[str]
    tags: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = Field(default="low", alias="riskLevel")


class EvalResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    case_id: str = Field(alias="caseId")
    suite: EvalSuite
    target: str
    mode: EvalMode
    metrics: dict[str, float]
    overall: float
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list, alias="failureReasons")
    bad_case_tags: list[str] = Field(default_factory=list, alias="badCaseTags")
    latency_ms: int | None = Field(default=None, alias="latencyMs")
    trace: AgentTrace


class EvalSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    suite: str
    mode: EvalMode
    total: int
    passed: int
    failed: int
    pass_rate: float = Field(alias="passRate")
    average_overall: float = Field(alias="averageOverall")
    min_pass_rate: float = Field(alias="minPassRate")
    min_overall: float = Field(alias="minOverall")
    generated_at: str = Field(alias="generatedAt")
    results: list[EvalResult]
