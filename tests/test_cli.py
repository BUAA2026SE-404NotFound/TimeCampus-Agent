from timecampus_agent.cli import parse_points


def test_parse_points() -> None:
    points = parse_points("Main Building,39.981,116.34;Library,39.982,116.341")

    assert len(points) == 2
    assert points[0].name == "Main Building"
    assert points[1].lng == 116.341
