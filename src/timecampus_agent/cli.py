from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence

from rich.console import Console

from timecampus_agent.agent import create_agent_executor
from timecampus_agent.backend import RoutePoint, TimeCampusBackendClient
from timecampus_agent.config import load_settings
from timecampus_agent.evaluation.embedding_benchmark import (
    handle_embedding_benchmark_command,
    register_embedding_benchmark_command,
)
from timecampus_agent.evaluation.cli import handle_eval_command, register_eval_commands
from timecampus_agent.mcp_client import call_timecampus_mcp_tool, list_timecampus_mcp_tool_names

console = Console()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="timecampus-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Run the Python agent.")
    ask_parser.add_argument("prompt")
    ask_parser.add_argument("--agent", choices=["auto", "operations", "guide"], default="auto")

    search_parser = subparsers.add_parser("rag-search", help="Search backend RAG context.")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=6)

    draft_parser = subparsers.add_parser("draft", help="Generate a backend grounded draft.")
    draft_parser.add_argument("task")
    draft_parser.add_argument("--limit", type=int, default=6)

    route_parser = subparsers.add_parser("route", help="Calculate a visitor walking route.")
    route_parser.add_argument("points", help="Semicolon-separated name,lat,lng points.")

    subparsers.add_parser("mcp-tools", help="List tools from the backend MCP server.")
    serve_parser = subparsers.add_parser("serve", help="Run the local Agent HTTP service.")
    serve_parser.add_argument("--host")
    serve_parser.add_argument("--port", type=int)
    register_eval_commands(subparsers)
    register_embedding_benchmark_command(subparsers)

    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "eval":
        return handle_eval_command(args, settings, console)

    if args.command == "embedding-benchmark":
        return handle_embedding_benchmark_command(args, console)

    if args.command == "serve":
        if not settings.agent_api_token:
            raise SystemExit("TIMECAMPUS_AGENT_API_TOKEN is required.")
        import uvicorn

        from timecampus_agent.service import create_app

        uvicorn.run(
            create_app(settings),
            host=args.host or settings.agent_api_host,
            port=args.port or settings.agent_api_port,
        )
        return 0

    if args.command == "ask":
        executor = create_agent_executor(settings, default_agent=args.agent)
        result = executor.invoke({"messages": [{"role": "user", "content": args.prompt}]})
        console.print(_extract_agent_output(result))
        return 0

    if args.command == "rag-search" and settings.mcp_token:
        result = asyncio.run(
            call_timecampus_mcp_tool(
                "timecampus_rag_search",
                {
                    "query": args.query,
                    "limit": args.limit,
                    "types": ["poi", "media", "comment", "guideline", "knowledge"],
                    "includePending": True,
                },
                settings,
            )
        )
        _print_json(_extract_mcp_tool_result(result))
        return 0

    client = TimeCampusBackendClient(settings.api_base_url, admin_token=settings.admin_token)
    if args.command == "draft" and not client.admin_token:
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


def _extract_mcp_tool_result(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        return result
    structured = result.get("structuredContent")
    if structured is not None:
        return structured
    content = result.get("content")
    if isinstance(content, list):
        texts = [
            item.get("text")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        ]
        if len(texts) == 1:
            try:
                return json.loads(str(texts[0]))
            except json.JSONDecodeError:
                return {"text": texts[0]}
        if texts:
            return {"text": "\n".join(str(text) for text in texts)}
    return result


if __name__ == "__main__":
    raise SystemExit(main())
