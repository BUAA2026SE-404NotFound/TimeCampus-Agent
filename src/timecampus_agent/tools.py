from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from timecampus_agent.backend import RoutePoint, TimeCampusBackendClient

ToolHandler = Callable[[dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        result = self.handler(arguments)
        if inspect.isawaitable(result):
            return await result
        return result

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


class RagSearchInput(BaseModel):
    query: str = Field(description="Maintenance or copy-editing query.")
    limit: int = Field(default=6, ge=1, le=20)
    types: list[str] = Field(
        default_factory=lambda: ["poi", "media", "comment", "guideline", "knowledge"]
    )
    poi_id: int | None = Field(default=None, alias="poiId")
    include_pending: bool = Field(default=True, alias="includePending")


class DraftInput(BaseModel):
    task: str = Field(description="Admin maintenance task to draft.")
    limit: int = Field(default=6, ge=1, le=20)
    types: list[str] = Field(default_factory=lambda: ["poi", "media", "guideline", "knowledge"])
    poi_id: int | None = Field(default=None, alias="poiId")
    include_pending: bool = Field(default=True, alias="includePending")


class WalkingRouteInput(BaseModel):
    points: list[RoutePoint] = Field(min_length=2, max_length=8)


class PublicPoiSearchInput(BaseModel):
    keyword: str | None = Field(default=None, description="Optional public POI keyword.")


def build_backend_tools(client: TimeCampusBackendClient) -> list[ToolSpec]:
    return [*build_operations_tools(client), *build_guide_tools(client)]


def build_operations_tools(client: TimeCampusBackendClient) -> list[ToolSpec]:
    async def rag_search(arguments: dict[str, Any]) -> str:
        data = RagSearchInput.model_validate(arguments)
        result = await asyncio.to_thread(
            client.rag_search,
            data.query,
            data.limit,
            data.types,
            data.poi_id,
            data.include_pending,
        )
        return _json(result)

    async def admin_draft(arguments: dict[str, Any]) -> str:
        data = DraftInput.model_validate(arguments)
        result = await asyncio.to_thread(
            client.agent_draft,
            data.task,
            data.limit,
            data.types,
            data.poi_id,
            data.include_pending,
        )
        return _json(result)

    return [
        ToolSpec(
            name="timecampus_rag_search",
            description="Search TimeCampus POI, media, comments, maintenance guidelines and knowledge documents.",
            parameters=RagSearchInput.model_json_schema(),
            handler=rag_search,
        ),
        ToolSpec(
            name="timecampus_admin_draft",
            description="Ask the backend to produce a grounded maintenance draft and quality gate.",
            parameters=DraftInput.model_json_schema(),
            handler=admin_draft,
        ),
    ]


def build_guide_tools(client: TimeCampusBackendClient) -> list[ToolSpec]:
    async def public_poi_search(arguments: dict[str, Any]) -> str:
        data = PublicPoiSearchInput.model_validate(arguments)
        result = await asyncio.to_thread(client.search_public_pois, data.keyword)
        return _json({"pois": result, "count": len(result)})

    async def walking_route(arguments: dict[str, Any]) -> str:
        data = WalkingRouteInput.model_validate(arguments)
        result = await asyncio.to_thread(client.walking_route, data.points)
        return _json(result)

    return [
        ToolSpec(
            name="timecampus_public_poi_search",
            description="Search public, published TimeCampus POIs before planning a visitor route.",
            parameters=PublicPoiSearchInput.model_json_schema(),
            handler=public_poi_search,
        ),
        ToolSpec(
            name="timecampus_walking_route",
            description="Calculate a public visitor walking route summary for 2-8 campus points.",
            parameters=WalkingRouteInput.model_json_schema(),
            handler=walking_route,
        ),
    ]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
