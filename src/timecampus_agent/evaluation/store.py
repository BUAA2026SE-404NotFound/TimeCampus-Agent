from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from timecampus_agent.evaluation.models import EvalSummary


class EvalStore:
    def __init__(self, root: Path, max_runs: int = 20) -> None:
        self.root = root
        self.runs_dir = root / "runs"
        self.bad_cases_file = root / "bad-cases.jsonl"
        self.max_runs = max_runs
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, summary: EvalSummary) -> None:
        with self._lock:
            self._write_json(self._run_path(summary.run_id), summary.model_dump(by_alias=True))
            stale = sorted(
                self.runs_dir.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[self.max_runs :]
            for path in stale:
                path.unlink(missing_ok=True)

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        runs = []
        with self._lock:
            for path in self.runs_dir.glob("*.json"):
                payload = self._read_json(path)
                if payload:
                    runs.append(_run_summary(payload))
        return sorted(runs, key=lambda item: item["generatedAt"], reverse=True)[:limit]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_json(self._run_path(run_id))

    def add_bad_case(self, run_id: str, case_id: str, note: str = "") -> dict[str, Any]:
        with self._lock:
            run = self.get_run(run_id)
            if not run:
                raise KeyError("Eval run not found.")
            result = next(
                (item for item in run.get("results", []) if item.get("caseId") == case_id),
                None,
            )
            if not result:
                raise KeyError("Eval result not found.")
            existing = next(
                (
                    item
                    for item in self.list_bad_cases("all")
                    if item["runId"] == run_id and item["caseId"] == case_id
                ),
                None,
            )
            if existing:
                return existing
            now = _now()
            item = {
                "id": str(uuid4()),
                "runId": run_id,
                "caseId": case_id,
                "suite": result.get("suite"),
                "status": "open",
                "note": note.strip(),
                "resolution": "",
                "failureReasons": result.get("failureReasons", []),
                "tags": result.get("badCaseTags", []),
                "trace": result.get("trace", {}),
                "createdAt": now,
                "updatedAt": now,
            }
            self._append_event({"event": "created", **item})
            return item

    def update_bad_case(
        self,
        bad_case_id: str,
        status: str,
        resolution: str = "",
    ) -> dict[str, Any]:
        _uuid(bad_case_id)
        if status not in {"open", "resolved"}:
            raise ValueError("Unsupported bad case status.")
        with self._lock:
            current = next(
                (item for item in self.list_bad_cases("all") if item["id"] == bad_case_id),
                None,
            )
            if not current:
                raise KeyError("Bad case not found.")
            event = {
                "event": "updated",
                "id": bad_case_id,
                "status": status,
                "resolution": resolution.strip(),
                "updatedAt": _now(),
            }
            self._append_event(event)
            updated = {**current, **event}
            updated.pop("event", None)
            return updated

    def list_bad_cases(self, status: str = "all") -> list[dict[str, Any]]:
        if status not in {"all", "open", "resolved"}:
            raise ValueError("Unsupported bad case status.")
        folded: dict[str, dict[str, Any]] = {}
        with self._lock:
            if self.bad_cases_file.exists():
                for line in self.bad_cases_file.read_text(encoding="utf-8").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    item_id = event.get("id")
                    if not item_id:
                        continue
                    folded[item_id] = {**folded.get(item_id, {}), **event}
        items = [
            {key: value for key, value in item.items() if key != "event"}
            for item in folded.values()
            if status == "all" or item.get("status") == status
        ]
        return sorted(items, key=lambda item: item.get("updatedAt", ""), reverse=True)

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{_uuid(run_id)}.json"

    def _append_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            with self.bad_cases_file.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exception:
        raise ValueError("Invalid id.") from exception


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "runId",
        "suite",
        "mode",
        "repetitions",
        "caseCount",
        "total",
        "passed",
        "failed",
        "passRate",
        "averageOverall",
        "metricAverages",
        "consistencyRate",
        "p50LatencyMs",
        "p95LatencyMs",
        "highRiskPassed",
        "gatePassed",
        "agentVersion",
        "gitCommit",
        "model",
        "promptVersion",
        "datasetVersion",
        "generatedAt",
    )
    return {key: payload.get(key) for key in keys}
