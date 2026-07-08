from timecampus_agent.llm import _stream_content


def test_stream_content_parses_openai_sse_delta() -> None:
    assert _stream_content('data: {"choices":[{"delta":{"content":"北航"}}]}') == "北航"
    assert _stream_content("data: [DONE]") is None
    assert _stream_content(": ping") is None
