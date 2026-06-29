from timecampus_agent.memory import SessionStore


def test_session_store_persists_history_and_recovers_valid_lines(tmp_path) -> None:
    store = SessionStore(tmp_path, history_limit=2)
    session = store.create()
    session_id = session["id"]

    store.append(session_id, "user", "先查询主楼资料")
    store.append(session_id, "assistant", "已查询主楼资料。")
    store.append(session_id, "user", "基于刚才结果继续检查历史影像")

    reloaded = SessionStore(tmp_path, history_limit=2)
    detail = reloaded.get(session_id)
    assert detail is not None
    assert detail["title"] == "先查询主楼资料"
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert reloaded.prompt_messages(session_id) == [
        {"role": "assistant", "content": "已查询主楼资料。"},
        {"role": "user", "content": "基于刚才结果继续检查历史影像"},
    ]
    assert not list((tmp_path / "sessions").glob("*.tmp"))


def test_session_store_rejects_invalid_session_path(tmp_path) -> None:
    store = SessionStore(tmp_path)

    try:
        store.get("../outside")
    except ValueError as exception:
        assert "Invalid session id" in str(exception)
    else:
        raise AssertionError("invalid session id should be rejected")
