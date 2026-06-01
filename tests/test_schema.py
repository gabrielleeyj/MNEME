from __future__ import annotations

from mneme.db import init_db


def test_schema_creates_events_and_facts_tables():
    conn = init_db(":memory:")
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"events", "facts"} <= tables


def test_append_only_triggers_exist():
    conn = init_db(":memory:")
    triggers = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
    }
    assert {"events_no_update", "events_no_delete"} <= triggers


def test_apply_schema_is_idempotent():
    conn = init_db(":memory:")
    # Re-applying must not raise (IF NOT EXISTS everywhere).
    from mneme.db import apply_schema

    apply_schema(conn)


def test_event_check_constraints_reject_bad_enums():
    import sqlite3

    conn = init_db(":memory:")
    try:
        conn.execute(
            "INSERT INTO events (ts, actor, type, content) VALUES (?,?,?,?)",
            ("2026-01-01T00:00:00+00:00", "robot", "message", "hi"),
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("CHECK constraint on actor was not enforced")
