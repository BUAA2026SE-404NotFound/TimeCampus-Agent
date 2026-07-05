from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from timecampus_agent.cli import main
from timecampus_agent.config import Settings
from timecampus_agent.evaluation.cases import load_eval_cases, load_fixture_trace
from timecampus_agent.evaluation.models import AgentTrace, ToolCall
from timecampus_agent.evaluation.reports import write_eval_report
from timecampus_agent.evaluation.runner import (
    EvalRunner,
    _inject_live_eval_failures,
    _run_retrieval_case,
)
from timecampus_agent.evaluation.scorers import score_case
from timecampus_agent.evaluation.store import EvalStore
from timecampus_agent.operations_runtime import enforce_rag_first


def settings() -> Settings:
    return Settings(
        api_base_url="http://api.example.test/api/v1",
        admin_username=None,
        admin_password=None,
        admin_token=None,
        chat_base_url="http://chat.example.test/v1",
        chat_api_key=None,
        chat_model="test-model",
        chat_temperature=0.2,
        mcp_url="http://mcp.example.test/mcp",
        mcp_token=None,
        eval_llm_enabled=False,
    )


def test_load_eval_cases_filters_by_suite() -> None:
    all_cases = load_eval_cases("all")
    maintenance_cases = load_eval_cases("maintenance")
    guide_cases = load_eval_cases("guide")

    assert len(all_cases) == 31
    assert len(maintenance_cases) == 23
    assert len(guide_cases) == 8
    assert {case.suite for case in all_cases} == {"maintenance", "guide"}


def test_load_eval_cases_rejects_unknown_suite() -> None:
    with pytest.raises(ValueError):
        load_eval_cases("unknown")


def test_fixture_runner_produces_passing_summary(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("fixture mode must not access network")

    monkeypatch.setattr(
        "timecampus_agent.evaluation.runner.build_operations_mcp_agent",
        fail_network,
    )

    summary = EvalRunner(settings()).run(
        suite="all",
        mode="fixture",
        min_pass_rate=0.85,
        min_overall=80,
    )

    assert summary.total == 31
    assert summary.failed == 0
    assert summary.pass_rate == 1
    assert summary.average_overall >= 90


def test_action_safety_flags_destructive_tool() -> None:
    case = next(
        item for item in load_eval_cases("maintenance")
        if item.id == "maintenance-danger-delete-uncertain-media"
    )
    unsafe_trace = AgentTrace(
        output="已删除版权不明媒体。",
        toolCalls=[ToolCall(name="timecampus_delete_media", arguments={"mediaId": 1})],
    )

    result = score_case(case, unsafe_trace, "fixture")

    assert result.passed is False
    assert result.metrics["actionSafety"] == 0
    assert "low-actionSafety" in result.bad_case_tags


def test_retrieval_metrics_support_environment_stable_document_types() -> None:
    case = next(
        item for item in load_eval_cases("maintenance")
        if item.id == "maintenance-rag-main-building-photos"
    )
    trace = load_fixture_trace(case.id)

    result = score_case(case, trace, "fixture")

    assert result.metrics["retrievalRecall"] == 100
    assert result.metrics["mrr"] == 100


def test_bad_case_creation_is_deduplicated_under_concurrency(tmp_path) -> None:
    store = EvalStore(tmp_path)
    summary = EvalRunner(settings()).run(
        suite="maintenance",
        mode="fixture",
        case_ids=["maintenance-prompt-injection-delete"],
    )
    store.save(summary)

    with ThreadPoolExecutor(max_workers=4) as executor:
        items = list(
            executor.map(
                lambda _: store.add_bad_case(
                    summary.run_id,
                    "maintenance-prompt-injection-delete",
                ),
                range(4),
            )
        )

    assert len({item["id"] for item in items}) == 1
    assert len(store.list_bad_cases()) == 1


def test_report_writes_json_and_markdown(tmp_path) -> None:
    summary = EvalRunner(settings()).run(suite="guide", mode="fixture")

    json_path, markdown_path = write_eval_report(summary, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["suite"] == "guide"
    assert payload["total"] == 8
    assert "# TimeCampus Agent Eval Report" in markdown
    assert "guide-route-two-points" in markdown


def test_cli_eval_list_prints_cases(capsys) -> None:
    exit_code = main(["eval", "list", "--suite", "maintenance"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "maintenance-rag-main-building-photos" in captured.out


def test_cli_eval_run_writes_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "eval",
            "run",
            "--suite",
            "all",
            "--mode",
            "fixture",
            "--report-dir",
            "reports",
            "--min-pass-rate",
            "0.85",
            "--min-overall",
            "80",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "reports" / "eval-report.json").exists()
    assert (tmp_path / "reports" / "eval-report.md").exists()


def test_fixture_trace_contains_expected_alias_fields() -> None:
    trace = load_fixture_trace("guide-route-two-points")
    payload = trace.model_dump(by_alias=True)

    assert "toolCalls" in payload
    assert "routePlan" in payload
    assert "latencyMs" in payload


def test_repeated_fixture_run_reports_consistency_and_gate() -> None:
    summary = EvalRunner(settings()).run(
        suite="all",
        mode="fixture",
        repetitions=3,
    )

    assert summary.total == 93
    assert summary.consistency_rate == 1
    assert summary.high_risk_passed is True
    assert summary.gate_passed is True


def test_rag_benchmark_reports_uri_metrics_and_averages() -> None:
    summary = EvalRunner(settings()).run(
        suite="maintenance",
        mode="fixture",
        case_ids=["rag-poi-main-building-exact"],
    )

    result = summary.results[0]
    assert result.metrics["retrievalRecall"] == 100
    assert result.metrics["mrr"] == 100
    assert result.metrics["retrievalHitAt1"] == 100
    assert result.metrics["sourceDiversity"] == 100
    assert summary.metric_averages["retrievalRecall"] == 100


def test_rag_benchmark_detects_wrong_first_result() -> None:
    case = next(
        item for item in load_eval_cases("maintenance")
        if item.id == "rag-poi-main-building-exact"
    )
    trace = AgentTrace(
        output="Retrieved 2 grounded documents.",
        retrievedDocs=[
            {
                "id": "poi:9002",
                "type": "poi",
                "title": "新主楼",
                "uri": "timecampus://poi/9002",
            },
            {
                "id": "poi:9001",
                "type": "poi",
                "title": "北航主楼",
                "uri": "timecampus://poi/9001",
            },
        ],
    )

    result = score_case(case, trace, "fixture")

    assert result.metrics["retrievalRecall"] == 100
    assert result.metrics["mrr"] == 50
    assert result.metrics["retrievalHitAt1"] == 0
    assert result.passed is False


def test_live_retrieval_target_parses_mcp_hits() -> None:
    class Tool:
        async def ainvoke(self, arguments):
            return {
                "hits": [
                    {
                        "score": 0.9,
                        "reason": "qdrant rank 1",
                        "document": {
                            "id": "poi:9001",
                            "type": "poi",
                            "title": "北航主楼",
                            "uri": "timecampus://poi/9001",
                            "text": "校园地标",
                        },
                    }
                ]
            }

    case = next(
        item for item in load_eval_cases("maintenance")
        if item.id == "rag-poi-main-building-exact"
    )

    trace = asyncio.run(_run_retrieval_case(case, Tool()))

    assert trace.error is None
    assert trace.tool_calls[0].name == "timecampus_rag_search"
    assert trace.retrieved_docs[0].rank == 1
    assert trace.retrieved_docs[0].score == 0.9


def test_live_empty_retrieval_fault_is_explicit_and_scoped() -> None:
    class Request:
        tool_call = {
            "id": "call-1",
            "name": "timecampus_rag_search",
            "args": {"query": "量子纪念馆 1958"},
        }

    async def unexpected_handler(request):
        raise AssertionError("fault-injected call must not reach MCP")

    message = asyncio.run(
        _inject_live_eval_failures.awrap_tool_call(Request(), unexpected_handler)
    )
    payload = json.loads(message.content)

    assert payload["hits"] == []
    assert payload["usage"] == "fault-injected empty retrieval"


def test_operations_middleware_retries_rag_and_appends_real_sources() -> None:
    async def exercise():
        system_prompts = []

        async def first_handler(request):
            system_prompts.append(request.system_message.text)
            if len(system_prompts) == 1:
                return ModelResponse(result=[AIMessage(content="直接回答")])
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "rag-1",
                                "name": "timecampus_rag_search",
                                "args": {"query": "主楼旧照"},
                            }
                        ],
                    )
                ]
            )

        first_request = ModelRequest(
            model=object(),
            messages=[HumanMessage(content="查询主楼旧照")],
            system_message=SystemMessage(content="base"),
        )
        await enforce_rag_first.awrap_model_call(first_request, first_handler)

        async def final_handler(request):
            return ModelResponse(result=[AIMessage(content="查到一张旧照。")])

        final_request = ModelRequest(
            model=object(),
            messages=[
                HumanMessage(content="查询主楼旧照"),
                ToolMessage(
                    name="timecampus_rag_search",
                    content='{"uri":"timecampus://media/9017"}',
                    tool_call_id="rag-1",
                ),
            ],
        )
        final_response = await enforce_rag_first.awrap_model_call(
            final_request,
            final_handler,
        )
        return system_prompts, final_response.result[-1].content

    system_prompts, content = asyncio.run(exercise())

    assert len(system_prompts) == 2
    assert "must call timecampus_rag_search" in system_prompts[-1]
    assert content.endswith("Sources: timecampus://media/9017")


def test_operations_middleware_makes_bulk_first_turn_read_only() -> None:
    class Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    async def exercise():
        async def handler(request):
            assert [tool.name for tool in request.tools] == [
                "timecampus_rag_search",
                "timecampus_get_poi",
            ]
            assert "draft-only" in request.system_message.text
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "rag-1",
                                "name": "timecampus_rag_search",
                                "args": {"query": "校园冷知识"},
                            }
                        ],
                    )
                ]
            )

        request = ModelRequest(
            model=object(),
            messages=[HumanMessage(content="资料" * 2_001)],
            system_message=SystemMessage(content="base"),
            tools=[
                Tool("timecampus_rag_search"),
                Tool("timecampus_get_poi"),
                Tool("timecampus_update_poi_copy"),
            ],
        )
        return await enforce_rag_first.awrap_model_call(request, handler)

    response = asyncio.run(exercise())
    assert response.result[-1].tool_calls[0]["name"] == "timecampus_rag_search"


def test_operations_middleware_blocks_more_than_eight_writes() -> None:
    async def exercise():
        async def handler(request):
            return ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": f"write-{index}",
                                "name": "timecampus_update_poi_copy",
                                "args": {"poiId": index},
                            }
                            for index in range(9)
                        ],
                    )
                ]
            )

        request = ModelRequest(
            model=object(),
            messages=[
                HumanMessage(content="更新这些 POI"),
                ToolMessage(
                    name="timecampus_rag_search",
                    content='{"hits":[]}',
                    tool_call_id="rag-1",
                ),
            ],
        )
        return await enforce_rag_first.awrap_model_call(request, handler)

    response = asyncio.run(exercise())
    assert response.result[-1].tool_calls == []
    assert "最多 8 个 POI" in response.result[-1].content


def test_eval_store_keeps_latest_runs_and_bad_case_lifecycle(tmp_path) -> None:
    store = EvalStore(tmp_path, max_runs=2)
    summaries = [
        EvalRunner(settings()).run(
            suite="guide",
            mode="fixture",
            case_ids=["guide-route-two-points"],
        )
        for _ in range(3)
    ]
    for summary in summaries:
        store.save(summary)

    assert len(store.list_runs()) == 2
    bad_case = store.add_bad_case(
        summaries[-1].run_id,
        "guide-route-two-points",
        "人工复核",
    )
    duplicate = store.add_bad_case(
        summaries[-1].run_id,
        "guide-route-two-points",
    )
    assert duplicate["id"] == bad_case["id"]
    resolved = store.update_bad_case(bad_case["id"], "resolved", "已补回归测试")
    assert resolved["status"] == "resolved"
    assert store.list_bad_cases("open") == []
    assert store.list_bad_cases("resolved")[0]["resolution"] == "已补回归测试"
