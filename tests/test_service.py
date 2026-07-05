from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx

from timecampus_agent.config import load_settings
from timecampus_agent.service import (
    OperationDecision,
    create_app,
    tool_policy,
)


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeRuntime:
    def __init__(self) -> None:
        self.session_id = "16cf781c-28d9-4f4a-9352-95e33df9067d"

    async def start(self, task: str, session_id: str | None = None) -> dict:
        return {
            "threadId": "thread-1",
            "sessionId": session_id or self.session_id,
            "status": "approval_required",
            "pendingActions": [{"name": "timecampus_update_poi_copy", "arguments": {}}],
            "toolEvents": [],
            "output": task,
        }

    async def resume(self, thread_id: str, decisions: list[OperationDecision]) -> dict:
        return {
            "threadId": thread_id,
            "status": "completed",
            "pendingActions": [],
            "toolEvents": [],
            "output": decisions[0].type,
        }

    def list_sessions(self) -> list[dict]:
        return [
            {
                "id": self.session_id,
                "title": "主楼维护",
                "preview": "查询主楼",
                "messageCount": 2,
                "createdAt": "2026-06-28T00:00:00+00:00",
                "updatedAt": "2026-06-28T00:01:00+00:00",
            }
        ]

    def create_session(self, title: str | None = None) -> dict:
        return {**self.list_sessions()[0], "title": title or "新运营会话"}

    def get_session(self, session_id: str) -> dict:
        return {
            **self.list_sessions()[0],
            "id": session_id,
            "messages": [
                {
                    "id": "message-1",
                    "role": "user",
                    "content": "查询主楼",
                    "createdAt": "2026-06-28T00:00:00+00:00",
                }
            ],
        }

    def record_message(self, session_id: str, role: str, content: str) -> dict:
        return {
            **self.get_session(session_id),
            "messages": [
                {
                    "id": "message-2",
                    "role": role,
                    "content": content,
                    "createdAt": "2026-06-28T00:02:00+00:00",
                }
            ],
        }

    async def stream_start(self, session_id: str, task: str):
        yield "delta", {"content": "已查询"}
        yield "result", {
            "threadId": "thread-2",
            "sessionId": session_id,
            "status": "completed",
            "pendingActions": [],
            "toolEvents": [],
            "output": f"已查询：{task}",
        }
        yield "done", {"status": "completed"}

    async def stream_resume(self, thread_id: str, decisions: list[OperationDecision]):
        yield "delta", {"content": decisions[0].type}
        yield "done", {"status": "completed"}


def test_tool_policy_blocks_deletes_and_interrupts_writes() -> None:
    tools, policy = tool_policy(
        [
            FakeTool("timecampus_get_poi"),
            FakeTool("timecampus_walking_route"),
            FakeTool("timecampus_update_poi_copy"),
            FakeTool("timecampus_delete_poi"),
        ]
    )

    assert [tool.name for tool in tools] == [
        "timecampus_get_poi",
        "timecampus_walking_route",
        "timecampus_update_poi_copy",
    ]
    assert policy["timecampus_get_poi"] is False
    assert policy["timecampus_walking_route"] is False
    assert policy["timecampus_update_poi_copy"]["allowed_decisions"] == [
        "approve",
        "reject",
    ]


def test_service_requires_token_and_supports_approval(tmp_path) -> None:
    settings = replace(
        load_settings(),
        agent_api_token="test-token",
        eval_report_dir=str(tmp_path),
    )
    async def check() -> None:
        async with _client(settings) as client:
            response = await client.post(
                "/internal/v1/operations/runs", json={"task": "更新主楼"}
            )
            assert response.status_code == 401

            headers = {"X-TimeCampus-Agent-Token": "test-token"}
            started = await client.post(
                "/internal/v1/operations/runs",
                headers=headers,
                json={"task": "更新主楼"},
            )
            assert started.json()["status"] == "approval_required"

            resumed = await client.post(
                "/internal/v1/operations/runs/thread-1/decisions",
                headers=headers,
                json={"decisions": [{"type": "reject", "message": "保留原文"}]},
            )
            assert resumed.json()["status"] == "completed"
            assert resumed.json()["output"] == "reject"

    asyncio.run(check())


def test_operation_task_accepts_bulk_input_and_rejects_over_limit(tmp_path) -> None:
    settings = replace(
        load_settings(),
        agent_api_token="test-token",
        eval_report_dir=str(tmp_path),
    )

    async def check() -> None:
        headers = {"X-TimeCampus-Agent-Token": "test-token"}
        async with _client(settings) as client:
            accepted = await client.post(
                "/internal/v1/operations/runs",
                headers=headers,
                json={"task": "校" * 7_500},
            )
            rejected = await client.post(
                "/internal/v1/operations/runs",
                headers=headers,
                json={"task": "校" * 20_001},
            )

        assert accepted.status_code == 200
        assert rejected.status_code == 422
        assert rejected.json()["detail"][0]["type"] == "string_too_long"

    asyncio.run(check())


def test_fixture_eval_api_writes_latest_report(tmp_path) -> None:
    settings = replace(
        load_settings(),
        agent_api_token="test-token",
        eval_report_dir=str(tmp_path),
    )
    async def check() -> None:
        async with _client(settings) as client:
            response = await client.post(
                "/internal/v1/evals/runs",
                headers={"X-TimeCampus-Agent-Token": "test-token"},
                json={"suite": "all", "mode": "fixture"},
            )
            assert response.status_code == 200
            assert response.json()["total"] == 31
            assert response.json()["passed"] == 31

    asyncio.run(check())
    assert (tmp_path / "eval-report.json").exists()


def test_session_api_lists_history_and_streams_sse(tmp_path) -> None:
    settings = replace(load_settings(), agent_api_token="test-token")

    async def check() -> None:
        headers = {"X-TimeCampus-Agent-Token": "test-token"}
        async with _client(settings) as client:
            sessions = await client.get("/internal/v1/operations/sessions", headers=headers)
            session_id = sessions.json()["sessions"][0]["id"]
            detail = await client.get(
                f"/internal/v1/operations/sessions/{session_id}",
                headers=headers,
            )
            assert detail.json()["messages"][0]["content"] == "查询主楼"

            async with client.stream(
                "POST",
                f"/internal/v1/operations/sessions/{session_id}/messages/stream",
                headers=headers,
                json={"task": "继续查询"},
            ) as response:
                payload = (await response.aread()).decode()
            assert response.status_code == 200
            assert "event: delta" in payload
            assert '"content": "已查询"' in payload
            assert "event: result" in payload

    asyncio.run(check())


def test_eval_stream_history_and_bad_case_api(tmp_path) -> None:
    settings = replace(
        load_settings(),
        agent_api_token="test-token",
        eval_report_dir=str(tmp_path),
    )

    async def check() -> None:
        headers = {"X-TimeCampus-Agent-Token": "test-token"}
        async with _client(settings) as client:
            async with client.stream(
                "POST",
                "/internal/v1/evals/runs/stream",
                headers=headers,
                json={
                    "suite": "guide",
                    "mode": "fixture",
                    "repetitions": 2,
                    "caseIds": ["guide-route-two-points"],
                },
            ) as response:
                payload = (await response.aread()).decode()
            assert "event: started" in payload
            assert payload.count("event: case") == 2
            assert "event: result" in payload

            history = await client.get("/internal/v1/evals/runs", headers=headers)
            run_id = history.json()["runs"][0]["runId"]
            detail = await client.get(
                f"/internal/v1/evals/runs/{run_id}",
                headers=headers,
            )
            assert detail.json()["repetitions"] == 2

            created = await client.post(
                "/internal/v1/evals/bad-cases",
                headers=headers,
                json={
                    "runId": run_id,
                    "caseId": "guide-route-two-points",
                    "note": "回归检查",
                },
            )
            bad_case_id = created.json()["id"]
            resolved = await client.patch(
                f"/internal/v1/evals/bad-cases/{bad_case_id}",
                headers=headers,
                json={"status": "resolved", "resolution": "已处理"},
            )
            assert resolved.json()["status"] == "resolved"
            bad_cases = await client.get(
                "/internal/v1/evals/bad-cases?status=resolved",
                headers=headers,
            )
            assert bad_cases.json()["badCases"][0]["resolution"] == "已处理"

    asyncio.run(check())


def _client(settings):
    app = create_app(settings, FakeRuntime())
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    )
