from __future__ import annotations

import json
from typing import Any

from timecampus_agent.config import Settings
from timecampus_agent.evaluation.models import AgentTrace, EvalCase
from timecampus_agent.llm import ChatClient


def score_with_llm_judge(settings: Settings, case: EvalCase, trace: AgentTrace) -> dict[str, float]:
    if not settings.eval_llm_enabled or not settings.chat_api_key:
        return {}
    llm = ChatClient(settings, temperature=0)
    prompt = {
        "case": case.model_dump(by_alias=True),
        "trace": {
            "output": trace.output,
            "toolCalls": [tool.model_dump() for tool in trace.tool_calls],
            "retrievedDocs": [doc.model_dump() for doc in trace.retrieved_docs],
            "routePlan": trace.route_plan,
            "error": trace.error,
        },
    }
    response = llm.complete_sync(
        [
            {
                "role": "system",
                "content": (
                    "You are an evaluator for TimeCampus agent traces. "
                    "Return strict JSON with numeric 0-100 scores for answerCorrectness "
                    "and faithfulness. Correctness measures whether the answer satisfies "
                    "the case expectation. Faithfulness measures whether every factual "
                    "claim is supported by retrievedDocs or successful read-only tool "
                    "results in toolCalls. Do not include explanations."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
    )
    content = response.get("content", "")
    value = _parse_json_object(str(content))
    return {
        f"llm{name[0].upper()}{name[1:]}": _clamp_score(raw)
        for name, raw in value.items()
        if name in {"answerCorrectness", "faithfulness"}
    }


def _parse_json_object(value: str) -> dict[str, Any]:
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        parsed = json.loads(value[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clamp_score(value: Any) -> float:
    if not isinstance(value, int | float):
        return 0
    return float(max(0, min(100, value)))
