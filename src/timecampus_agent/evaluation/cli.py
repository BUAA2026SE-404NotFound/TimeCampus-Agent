from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from rich.console import Console

from timecampus_agent.config import Settings
from timecampus_agent.evaluation.cases import case_catalog
from timecampus_agent.evaluation.reports import write_eval_report
from timecampus_agent.evaluation.runner import EvalRunner


def register_eval_commands(subparsers) -> None:
    eval_parser = subparsers.add_parser("eval", help="Run TimeCampus agent evaluations.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)

    list_parser = eval_subparsers.add_parser("list", help="List built-in eval cases.")
    list_parser.add_argument("--suite", choices=["all", "maintenance", "guide"], default="all")

    run_parser = eval_subparsers.add_parser("run", help="Run an eval suite and write reports.")
    run_parser.add_argument("--suite", choices=["all", "maintenance", "guide"], default="all")
    run_parser.add_argument("--mode", choices=["fixture", "live"], default="fixture")
    run_parser.add_argument("--report-dir", default="eval-reports")
    run_parser.add_argument("--min-pass-rate", type=float, default=0.85)
    run_parser.add_argument("--min-overall", type=float, default=80)
    run_parser.add_argument("--min-consistency", type=float, default=0.8)
    run_parser.add_argument("--repetitions", type=int, choices=range(1, 6), default=1)
    run_parser.add_argument("--case-id", action="append", dest="case_ids")


def handle_eval_command(args: Namespace, settings: Settings, console: Console) -> int:
    if args.eval_command == "list":
        console.print_json(json.dumps({"cases": case_catalog(args.suite)}, ensure_ascii=False))
        return 0

    if args.eval_command == "run":
        runner = EvalRunner(settings)
        summary = runner.run(
            suite=args.suite,
            mode=args.mode,
            min_pass_rate=args.min_pass_rate,
            min_overall=args.min_overall,
            min_consistency=args.min_consistency,
            repetitions=args.repetitions,
            case_ids=args.case_ids,
        )
        json_path, markdown_path = write_eval_report(summary, Path(args.report_dir))
        console.print_json(
            json.dumps(
                {
                    "suite": summary.suite,
                    "mode": summary.mode,
                    "total": summary.total,
                    "passed": summary.passed,
                    "failed": summary.failed,
                    "passRate": summary.pass_rate,
                    "averageOverall": summary.average_overall,
                    "consistencyRate": summary.consistency_rate,
                    "p95LatencyMs": summary.p95_latency_ms,
                    "gatePassed": summary.gate_passed,
                    "jsonReport": str(json_path),
                    "markdownReport": str(markdown_path),
                },
                ensure_ascii=False,
            )
        )
        return 0 if summary.gate_passed else 1

    raise ValueError(f"Unknown eval command: {args.eval_command}")
