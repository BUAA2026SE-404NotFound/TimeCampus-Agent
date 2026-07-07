from timecampus_agent.agent import create_timecampus_agent, route_agent
from timecampus_agent.backend import TimeCampusBackendClient


class FakeModel:
    async def complete(self, messages, tools=None):
        return {"role": "assistant", "content": "ok"}


def test_route_agent_detects_guide_intent() -> None:
    agent, reason = route_agent("主楼到图书馆怎么走？")

    assert agent == "guide"
    assert "guide" in reason


def test_route_agent_defaults_to_operations() -> None:
    agent, reason = route_agent("检索主楼旧照并生成维护计划")

    assert agent == "operations"
    assert "operations" in reason


def test_timecampus_agent_builds_two_python_executors() -> None:
    agent = create_timecampus_agent(
        FakeModel(),
        TimeCampusBackendClient("http://api.example.test/api/v1"),
    )

    assert agent.operations_agent.name == "operations"
    assert agent.guide_agent.name == "guide"
