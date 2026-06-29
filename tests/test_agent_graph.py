from langchain_deepseek import ChatDeepSeek

from timecampus_agent.agent import create_timecampus_graph, route_agent
from timecampus_agent.backend import TimeCampusBackendClient


def test_route_agent_detects_guide_intent() -> None:
    agent, reason = route_agent("主楼到图书馆怎么走？")

    assert agent == "guide"
    assert "guide" in reason


def test_route_agent_defaults_to_operations() -> None:
    agent, reason = route_agent("检索主楼旧照并生成维护计划")

    assert agent == "operations"
    assert "operations" in reason


def test_timecampus_graph_compiles_supervisor_nodes() -> None:
    llm = ChatDeepSeek(api_key="test", base_url="http://chat.example.test/v1", model="test")
    graph = create_timecampus_graph(llm, TimeCampusBackendClient("http://api.example.test/api/v1"))

    assert {"supervisor", "operations_agent", "guide_agent"} <= set(graph.get_graph().nodes)
