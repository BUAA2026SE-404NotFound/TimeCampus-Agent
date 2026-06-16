from __future__ import annotations

import json

import pytest

from timecampus_agent.cli import main
from timecampus_agent.config import Settings
from timecampus_agent.evaluation.cases import load_eval_cases, load_fixture_trace
from timecampus_agent.evaluation.models import AgentTrace, ToolCall
from timecampus_agent.evaluation.reports import write_eval_report
from timecampus_agent.evaluation.runner import EvalRunner
from timecampus_agent.evaluation.scorers import score_case


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

    assert len(all_cases) == 9
    assert len(maintenance_cases) == 4
    assert len(guide_cases) == 5
    assert {case.suite for case in all_cases} == {"maintenance", "guide"}


def test_load_eval_cases_rejects_unknown_suite() -> None:
    with pytest.raises(ValueError):
        load_eval_cases("unknown")


def test_fixture_runner_produces_passing_summary(monkeypatch) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("fixture mode must not access network")

    monkeypatch.setattr(
        "timecampus_agent.evaluation.runner.call_timecampus_mcp_tool",
        fail_network,
    )

    summary = EvalRunner(settings()).run(
        suite="all",
        mode="fixture",
        min_pass_rate=0.85,
        min_overall=80,
    )

    assert summary.total == 9
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


def test_report_writes_json_and_markdown(tmp_path) -> None:
    summary = EvalRunner(settings()).run(suite="guide", mode="fixture")

    json_path, markdown_path = write_eval_report(summary, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["suite"] == "guide"
    assert payload["total"] == 5
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
