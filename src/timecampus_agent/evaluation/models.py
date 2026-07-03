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
    status: Literal["requested", "executed", "interrupted", "error"] = "executed"


class RetrievedDoc(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: str
    title: str
    uri: str
    text: str = ""
    rank: int | None = None
    score: float | None = None
    reason: str | None = None
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
    attempt: int = 1
    metrics: dict[str, float]
    overall: float
    passed: bool
    failure_reasons: list[str] = Field(default_factory=list, alias="failureReasons")
    bad_case_tags: list[str] = Field(default_factory=list, alias="badCaseTags")
    latency_ms: int | None = Field(default=None, alias="latencyMs")
    trace: AgentTrace


class EvalSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    suite: str
    mode: EvalMode
    repetitions: int
    case_count: int = Field(alias="caseCount")
    total: int
    passed: int
    failed: int
    pass_rate: float = Field(alias="passRate")
    average_overall: float = Field(alias="averageOverall")
    metric_averages: dict[str, float] = Field(alias="metricAverages")
    min_pass_rate: float = Field(alias="minPassRate")
    min_overall: float = Field(alias="minOverall")
    min_consistency: float = Field(alias="minConsistency")
    consistency_rate: float = Field(alias="consistencyRate")
    p50_latency_ms: int | None = Field(default=None, alias="p50LatencyMs")
    p95_latency_ms: int | None = Field(default=None, alias="p95LatencyMs")
    high_risk_passed: bool = Field(alias="highRiskPassed")
    gate_passed: bool = Field(alias="gatePassed")
    agent_version: str = Field(alias="agentVersion")
    git_commit: str = Field(alias="gitCommit")
    model: str
    prompt_version: str = Field(alias="promptVersion")
    dataset_version: str = Field(alias="datasetVersion")
    generated_at: str = Field(alias="generatedAt")
    results: list[EvalResult]
