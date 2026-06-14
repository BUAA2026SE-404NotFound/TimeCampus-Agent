from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from rich.console import Console

from timecampus_agent.agent import create_agent_executor
from timecampus_agent.backend import RoutePoint, TimeCampusBackendClient
from timecampus_agent.config import load_settings
from timecampus_agent.mcp_client import list_timecampus_mcp_tool_names

console = Console()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="timecampus-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run the LangChain agent.")
    ask_parser.add_argument("prompt")

    search_parser = subparsers.add_parser("rag-search", help="Search backend RAG context.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=6)

    draft_parser = subparsers.add_parser("draft", help="Generate a backend grounded draft.")
    draft_parser.add_argument("task")
    draft_parser.add_argument("--limit", type=int, default=6)

    route_parser = subparsers.add_parser("route", help="Calculate a visitor walking route.")
    route_parser.add_argument("points", help="Semicolon-separated name,lat,lng points.")

    subparsers.add_parser("mcp-tools", help="List tools from the backend MCP server.")

    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "ask":
        executor = create_agent_executor(settings)
        result = executor.invoke({"messages": [{"role": "user", "content": args.prompt}]})
        console.print(_extract_agent_output(result))
        return 0

    client = TimeCampusBackendClient(settings.api_base_url, admin_token=settings.admin_token)
    if args.command in {"rag-search", "draft"} and not client.admin_token:
        if settings.admin_username and settings.admin_password:
            client.login(settings.admin_username, settings.admin_password)
        else:
            raise SystemExit("Admin token or username/password is required.")

    if args.command == "rag-search":
        _print_json(client.rag_search(args.query, limit=args.limit))
    elif args.command == "draft":
        _print_json(client.agent_draft(args.task, limit=args.limit))
    elif args.command == "route":
        _print_json(client.walking_route(parse_points(args.points)))
    elif args.command == "mcp-tools":
        tool_names = asyncio.run(list_timecampus_mcp_tool_names(settings))
        _print_json({"tools": tool_names, "count": len(tool_names)})
    return 0


def parse_points(value: str) -> list[RoutePoint]:
    points: list[RoutePoint] = []
    for item in value.split(";"):
        fields = [field.strip() for field in item.split(",")]
        if len(fields) != 3:
            raise ValueError("Each point must be formatted as name,lat,lng")
        name, lat, lng = fields
        points.append(RoutePoint(name=name, lat=float(lat), lng=float(lng)))
    if len(points) < 2:
        raise ValueError("At least two points are required")
    return points


def _print_json(value: object) -> None:
    console.print_json(json.dumps(value, ensure_ascii=False))


def _extract_agent_output(result: object) -> object:
    if not isinstance(result, dict):
        return result
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        return result
    last_message = messages[-1]
    content = getattr(last_message, "content", None)
    if content is not None:
        return content
    if isinstance(last_message, dict):
        return last_message.get("content", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
