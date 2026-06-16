from __future__ import annotations

import json
from pathlib import Path

from timecampus_agent.evaluation.models import EvalSummary


def write_eval_report(summary: EvalSummary, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "eval-report.json"
    markdown_path = report_dir / "eval-report.md"
    json_path.write_text(
        json.dumps(summary.model_dump(by_alias=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(summary), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(summary: EvalSummary) -> str:
    lines = [
        "# TimeCampus Agent Eval Report",
        "",
        f"- suite: `{summary.suite}`",
        f"- mode: `{summary.mode}`",
        f"- generatedAt: `{summary.generated_at}`",
        f"- total: `{summary.total}`",
        f"- passed: `{summary.passed}`",
        f"- failed: `{summary.failed}`",
        f"- passRate: `{summary.pass_rate:.2%}`",
        f"- averageOverall: `{summary.average_overall:.2f}`",
        f"- gate: passRate >= `{summary.min_pass_rate:.2%}`, averageOverall >= `{summary.min_overall:.0f}`",
        "",
        "## Cases",
        "",
        "| Case | Suite | Overall | Passed | Bad Case Tags |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for result in summary.results:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(result.case_id),
                    result.suite,
                    f"{result.overall:.2f}",
                    "yes" if result.passed else "no",
                    _escape(", ".join(result.bad_case_tags) or "-"),
                ]
            )
            + " |"
        )
    failures = [result for result in summary.results if not result.passed]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("No failed cases.")
    else:
        for result in failures:
            lines.append(f"### {result.case_id}")
            lines.append("")
            lines.append(f"- overall: `{result.overall:.2f}`")
            lines.append(f"- reasons: {', '.join(result.failure_reasons)}")
            lines.append(f"- suggested bad-case tags: {', '.join(result.bad_case_tags) or '-'}")
            lines.append("")
    lines.extend(
        [
            "",
            "## Bad Case Loop",
            "",
            "- Treat failed cases as candidates for the tracked eval dataset only after human review.",
            "- Add a regression case when the failure exposes a reusable product risk.",
            "- Keep live-mode failures separate from fixture-mode failures when the backend or map provider is unavailable.",
            "",
        ]
    )
    return "\n".join(lines)


def _escape(value: str) -> str:
    return value.replace("|", "\\|")
