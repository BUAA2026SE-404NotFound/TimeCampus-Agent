from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import httpx
from pydantic import BaseModel, Field


class BackendError(RuntimeError):
    """Raised when the TimeCampus backend returns a failed response."""


class RoutePoint(BaseModel):
    name: str
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)


class TimeCampusBackendClient:
    def __init__(
        self,
        base_url: str,
        admin_token: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.admin_token = admin_token
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def login(self, admin_name: str, password: str) -> str:
        data = self._post("/admin/login", {"adminName": admin_name, "password": password})
        token = data.get("token")
        if not token:
            raise BackendError("Admin login succeeded but no token was returned")
        self.admin_token = str(token)
        return self.admin_token

    def rag_search(
        self,
        query: str,
        limit: int = 6,
        types: Iterable[str] | None = None,
        poi_id: int | None = None,
        include_pending: bool = True,
    ) -> dict[str, Any]:
        return self._post_admin(
            "/admin/agent/rag/search",
            {
                "query": query,
                "limit": limit,
                "types": list(types or ["poi", "media", "comment", "guideline", "knowledge"]),
                "poiId": poi_id,
                "includePending": include_pending,
            },
        )

    def agent_draft(
        self,
        task: str,
        limit: int = 6,
        types: Iterable[str] | None = None,
        poi_id: int | None = None,
        include_pending: bool = True,
    ) -> dict[str, Any]:
        return self._post_admin(
            "/admin/agent/draft",
            {
                "task": task,
                "limit": limit,
                "types": list(types or ["poi", "media", "guideline", "knowledge"]),
                "poiId": poi_id,
                "includePending": include_pending,
            },
        )

    def walking_route(self, points: Iterable[RoutePoint]) -> dict[str, Any]:
        return self._post("/map/walking-route", {"points": [point.model_dump() for point in points]})

    def search_public_pois(self, keyword: str | None = None) -> list[dict[str, Any]]:
        response = self._client.get(
            f"{self.base_url}/pois",
            params={"keyword": keyword} if keyword else None,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        data = self._unwrap_value(response.json(), "/pois")
        if not isinstance(data, list):
            raise BackendError("/pois returned non-list data")
        return [item for item in data if isinstance(item, dict)]

    def _post_admin(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.admin_token:
            raise BackendError("Admin token is required. Login first or set TIMECAMPUS_ADMIN_TOKEN.")
        return self._post(path, payload, token=self.admin_token)

    def _post(self, path: str, payload: dict[str, Any], token: str | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        response = self._client.post(f"{self.base_url}{path}", json=payload, headers=headers)
        response.raise_for_status()
        return self._unwrap(response.json(), path)

    def _unwrap(self, payload: Any, path: str) -> dict[str, Any]:
        data = self._unwrap_value(payload, path)
        if not isinstance(data, dict):
            raise BackendError(f"{path} returned non-object data")
        return data

    def _unwrap_value(self, payload: Any, path: str) -> Any:
        if not isinstance(payload, dict):
            raise BackendError(f"{path} returned a non-object response")
        if "code" in payload and payload["code"] != 0:
            raise BackendError(f"{path} failed: code={payload.get('code')} message={payload.get('message')}")
        return payload.get("data", payload)
