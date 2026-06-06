from __future__ import annotations

import json
from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from timecampus_agent.backend import RoutePoint, TimeCampusBackendClient


class RagSearchInput(BaseModel):
    query: str = Field(description="Maintenance or copy-editing query.")
    limit: int = Field(default=6, ge=1, le=20)
    types: list[str] = Field(default_factory=lambda: ["poi", "media", "comment", "guideline"])
    poi_id: int | None = Field(default=None)
    include_pending: bool = Field(default=True)


class DraftInput(BaseModel):
    task: str = Field(description="Admin maintenance task to draft.")
    limit: int = Field(default=6, ge=1, le=20)
    types: list[str] = Field(default_factory=lambda: ["poi", "media", "guideline"])
    poi_id: int | None = Field(default=None)
    include_pending: bool = Field(default=True)


class WalkingRouteInput(BaseModel):
    points: list[RoutePoint] = Field(min_length=2, max_length=8)


def build_backend_tools(client: TimeCampusBackendClient) -> list[StructuredTool]:
    def rag_search(
        query: Annotated[str, "Maintenance or copy-editing query."],
        limit: int = 6,
        types: list[str] | None = None,
        poi_id: int | None = None,
        include_pending: bool = True,
    ) -> str:
        result = client.rag_search(query, limit, types, poi_id, include_pending)
        return _json(result)

    def admin_draft(
        task: Annotated[str, "Admin maintenance task to draft."],
        limit: int = 6,
        types: list[str] | None = None,
        poi_id: int | None = None,
        include_pending: bool = True,
    ) -> str:
        result = client.agent_draft(task, limit, types, poi_id, include_pending)
        return _json(result)

    def walking_route(points: list[RoutePoint]) -> str:
        result = client.walking_route(points)
        return _json(result)

    return [
        StructuredTool.from_function(
            name="timecampus_rag_search",
            description="Search TimeCampus POI, media, comments and maintenance guidelines.",
            func=rag_search,
            args_schema=RagSearchInput,
        ),
        StructuredTool.from_function(
            name="timecampus_admin_draft",
            description="Ask the backend to produce a grounded maintenance draft and quality gate.",
            func=admin_draft,
            args_schema=DraftInput,
        ),
        StructuredTool.from_function(
            name="timecampus_walking_route",
            description="Calculate a public visitor walking route summary for 2-8 campus points.",
            func=walking_route,
            args_schema=WalkingRouteInput,
        ),
    ]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
