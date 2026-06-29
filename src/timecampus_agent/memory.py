from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

MAX_TITLE_LENGTH = 60
MAX_MEMORY_CHARS = 8_000


class SessionStore:
    """Small durable store for operator conversations."""

    def __init__(self, root: Path, history_limit: int = 40) -> None:
        self.root = root
        self.sessions_dir = root / "sessions"
        self.memory_file = root / "MEMORY.md"
        self.history_limit = max(2, history_limit)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory_file.touch(exist_ok=True)
        self._lock = threading.RLock()

    def create(self, title: str | None = None) -> dict[str, Any]:
        now = _now()
        session = {
            "id": str(uuid4()),
            "title": _title(title) or "新运营会话",
            "createdAt": now,
            "updatedAt": now,
            "messages": [],
        }
        with self._lock:
            self._write(session)
        return self._summary(session)

    def list(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        with self._lock:
            for path in self.sessions_dir.glob("*.jsonl"):
                session = self._read_path(path)
                if session:
                    sessions.append(self._summary(session))
        return sorted(sessions, key=lambda item: item["updatedAt"], reverse=True)

    def summary(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        if not session:
            raise KeyError(session_id)
        return self._summary(session)

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_path(self._path(session_id))

    def append(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported session message role.")
        normalized = content.strip()
        if not normalized:
            raise ValueError("Session message cannot be blank.")
        with self._lock:
            session = self._read_path(self._path(session_id))
            if not session:
                raise KeyError(session_id)
            timestamp = _now()
            session["messages"].append(
                {
                    "id": str(uuid4()),
                    "role": role,
                    "content": normalized,
                    "createdAt": timestamp,
                }
            )
            if role == "user" and not session["messages"][:-1]:
                session["title"] = _title(normalized) or session["title"]
            session["updatedAt"] = timestamp
            self._write(session)
            return session

    def prompt_messages(self, session_id: str) -> list[dict[str, str]]:
        session = self.get(session_id)
        if not session:
            raise KeyError(session_id)
        return [
            {"role": message["role"], "content": message["content"]}
            for message in session["messages"][-self.history_limit :]
        ]

    def memory_context(self) -> str:
        try:
            return self.memory_file.read_text(encoding="utf-8")[:MAX_MEMORY_CHARS].strip()
        except OSError:
            return ""

    def _path(self, session_id: str) -> Path:
        try:
            normalized = str(UUID(session_id))
        except (ValueError, AttributeError) as exception:
            raise ValueError("Invalid session id.") from exception
        return self.sessions_dir / f"{normalized}.jsonl"

    def _read_path(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        metadata: dict[str, Any] | None = None
        messages: list[dict[str, Any]] = []
        try:
            with path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if item.get("_type") == "metadata":
                        metadata = item
                    elif (
                        item.get("role") in {"user", "assistant"}
                        and isinstance(item.get("content"), str)
                    ):
                        messages.append(item)
        except OSError:
            return None
        if not metadata:
            return None
        return {
            "id": metadata["id"],
            "title": metadata.get("title") or "新运营会话",
            "createdAt": metadata.get("createdAt") or _now(),
            "updatedAt": metadata.get("updatedAt") or _now(),
            "messages": messages,
        }

    def _write(self, session: dict[str, Any]) -> None:
        path = self._path(session["id"])
        temporary = path.with_suffix(".jsonl.tmp")
        metadata = {
            "_type": "metadata",
            "id": session["id"],
            "title": session["title"],
            "createdAt": session["createdAt"],
            "updatedAt": session["updatedAt"],
        }
        try:
            with temporary.open("w", encoding="utf-8") as file:
                file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                for message in session["messages"]:
                    file.write(json.dumps(message, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _summary(session: dict[str, Any]) -> dict[str, Any]:
        messages = session["messages"]
        preview = messages[-1]["content"] if messages else ""
        return {
            "id": session["id"],
            "title": session["title"],
            "preview": re.sub(r"\s+", " ", preview).strip()[:120],
            "messageCount": len(messages),
            "createdAt": session["createdAt"],
            "updatedAt": session["updatedAt"],
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _title(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:MAX_TITLE_LENGTH]
