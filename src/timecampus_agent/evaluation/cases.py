from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from timecampus_agent.evaluation.models import AgentTrace, EvalCase

ALL_SUITES = {"maintenance", "guide"}
DATASET_VERSION = "2026-07-07.1"
DATASET_PATH = Path(__file__).with_name("cases.jsonl")


@lru_cache(maxsize=1)
def _records() -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(
        DATASET_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            case = EvalCase(**value["case"])
            trace = AgentTrace(**value["fixtureTrace"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exception:
            raise ValueError(f"Invalid eval dataset line {line_number}: {exception}") from exception
        records.append(
            {
                "case": case.model_dump(by_alias=True),
                "fixtureTrace": trace.model_dump(by_alias=True),
            }
        )
    return tuple(records)


def load_eval_cases(
    suite: str = "all",
    case_ids: list[str] | None = None,
) -> list[EvalCase]:
    if suite != "all" and suite not in ALL_SUITES:
        raise ValueError(f"Unknown eval suite: {suite}")
    wanted = set(case_ids or [])
    cases = [EvalCase(**record["case"]) for record in _records()]
    if suite != "all":
        cases = [case for case in cases if case.suite == suite]
    if wanted:
        cases = [case for case in cases if case.id in wanted]
        missing = wanted - {case.id for case in cases}
        if missing:
            raise ValueError(f"Unknown eval case ids: {', '.join(sorted(missing))}")
    return cases


def load_fixture_trace(case_id: str) -> AgentTrace:
    for record in _records():
        if record["case"]["id"] == case_id:
            return AgentTrace(**record["fixtureTrace"])
    raise ValueError(f"No fixture trace for eval case: {case_id}")


def case_catalog(suite: str = "all") -> list[dict[str, Any]]:
    return [
        {
            "id": case.id,
            "suite": case.suite,
            "target": case.target,
            "riskLevel": case.risk_level,
            "tags": case.tags,
            "checks": case.checks,
        }
        for case in load_eval_cases(suite)
    ]
