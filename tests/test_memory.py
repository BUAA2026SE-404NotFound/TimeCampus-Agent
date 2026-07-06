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


def test_session_store_does_not_list_empty_sessions(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create()

    assert store.list() == []

    store.append(session["id"], "user", "检索主楼资料")
    assert store.list() == []

    store.append(session["id"], "assistant", "已完成主楼资料检索")
    assert [item["id"] for item in store.list()] == [session["id"]]


def test_session_store_updates_generated_title(tmp_path) -> None:
    store = SessionStore(tmp_path)
    session = store.create()
    store.append(session["id"], "user", "一段非常长的运营任务正文")

    store.set_title(session["id"], "主楼冷知识整理")

    assert store.get(session["id"])["title"] == "主楼冷知识整理"


def test_session_store_persists_pending_runs(tmp_path) -> None:
    store = SessionStore(tmp_path)
    pending = {
        "session-1": {
            "threadId": "thread-1",
            "status": "approval_required",
            "pendingActions": [],
        }
    }

    store.save_pending_runs(pending)

    assert SessionStore(tmp_path).load_pending_runs() == pending
