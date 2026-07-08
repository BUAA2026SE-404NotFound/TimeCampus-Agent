from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from timecampus_agent.config import Settings


class ChatClient:
    def __init__(
        self,
        settings: Settings,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not settings.chat_api_key:
            raise RuntimeError("TIMECAMPUS_CHAT_API_KEY is required.")
        self.url = settings.chat_base_url.rstrip("/") + "/chat/completions"
        self.api_key = settings.chat_api_key
        self.model = settings.chat_model
        self.temperature = settings.chat_temperature if temperature is None else temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = self._payload(messages, tools)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=self._headers(), json=payload)
            response.raise_for_status()
            return _message(response.json())

    async def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        payload = {**self._payload(messages, tools), "stream": True}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self.url, headers=self._headers(), json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = _stream_content(line)
                    if chunk:
                        yield chunk

    def complete_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        response = httpx.post(
            self.url,
            headers=self._headers(),
            json=self._payload(messages, tools),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return _message(response.json())

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Chat completion returned no choices.")
    message = choices[0].get("message", {})
    if not isinstance(message, dict):
        raise RuntimeError("Chat completion returned an invalid message.")
    return _normalize_message(message)


def _stream_content(line: str) -> str | None:
    if not line.startswith("data:"):
        return None
    data = line.removeprefix("data:").strip()
    if not data or data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return None
    content = delta.get("content")
    return content if isinstance(content, str) else None


def _normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "role": message.get("role") or "assistant",
        "content": message.get("content") or "",
    }
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        normalized["tool_calls"] = [
            call for call in tool_calls if isinstance(call, dict)
        ]
    return normalized
