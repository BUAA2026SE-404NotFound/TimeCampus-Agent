from timecampus_agent.cli import _extract_mcp_tool_result, parse_points


def test_parse_points() -> None:
    points = parse_points("Main Building,39.981,116.34;Library,39.982,116.341")

    assert len(points) == 2
    assert points[0].name == "Main Building"
    assert points[1].lng == 116.341


def test_extract_mcp_tool_structured_content() -> None:
    payload = {
        "result": {
            "structuredContent": {
                "query": "主楼旧照",
                "hits": [{"id": "media:1"}],
            }
        }
    }

    assert _extract_mcp_tool_result(payload) == {
        "query": "主楼旧照",
        "hits": [{"id": "media:1"}],
    }


def test_extract_mcp_tool_text_json_content() -> None:
    payload = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": '{"query": "主楼旧照", "hits": []}',
                }
            ]
        }
    }

    assert _extract_mcp_tool_result(payload) == {"query": "主楼旧照", "hits": []}
