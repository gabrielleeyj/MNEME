from __future__ import annotations

import sqlite3

import pytest

from mneme.domain.events import Actor, EventType


def test_append_returns_event_with_assigned_id(log, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "hello", ts=at(1))
    assert event.event_id == 1
    assert event.actor is Actor.USER
    assert event.type is EventType.MESSAGE
    assert event.content == "hello"
    assert event.ts == at(1)


def test_append_accepts_string_enums(log, at):
    event = log.append("assistant", "reflection", "thinking", ts=at(1))
    assert event.actor is Actor.ASSISTANT
    assert event.type is EventType.REFLECTION


def test_replay_yields_events_in_append_order(log, at):
    log.append(Actor.USER, EventType.MESSAGE, "first", ts=at(1))
    log.append(Actor.ASSISTANT, EventType.MESSAGE, "second", ts=at(2))
    log.append(Actor.USER, EventType.MESSAGE, "third", ts=at(3))

    contents = [event.content for event in log.replay()]
    assert contents == ["first", "second", "third"]


def test_get_round_trips_event(log, at):
    appended = log.append(Actor.TOOL, EventType.TOOL_CALL, "ran", ts=at(5))
    fetched = log.get(appended.event_id)
    assert fetched == appended


def test_get_missing_event_raises_keyerror(log):
    with pytest.raises(KeyError):
        log.get(999)


def test_empty_content_is_rejected(log, at):
    with pytest.raises(ValueError):
        log.append(Actor.USER, EventType.MESSAGE, "", ts=at(1))


def test_invalid_actor_is_rejected(log, at):
    with pytest.raises(ValueError):
        log.append("robot", EventType.MESSAGE, "hi", ts=at(1))


def test_parent_event_id_must_reference_existing_event(log, at):
    with pytest.raises(sqlite3.IntegrityError):
        log.append(
            Actor.USER, EventType.MESSAGE, "orphan", ts=at(1), parent_event_id=42
        )


def test_parent_link_is_preserved(log, at):
    root = log.append(Actor.USER, EventType.MESSAGE, "root", ts=at(1))
    child = log.append(
        Actor.ASSISTANT,
        EventType.MESSAGE,
        "child",
        ts=at(2),
        parent_event_id=root.event_id,
    )
    assert child.parent_event_id == root.event_id


def test_events_cannot_be_updated(log, conn, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "immutable", ts=at(1))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE events SET content = 'tampered' WHERE event_id = ?",
            (event.event_id,),
        )
    assert log.get(event.event_id).content == "immutable"


def test_events_cannot_be_deleted(log, conn, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "permanent", ts=at(1))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM events WHERE event_id = ?", (event.event_id,))
    assert len(log) == 1


def test_len_counts_appended_events(log, at):
    assert len(log) == 0
    log.append(Actor.USER, EventType.MESSAGE, "a", ts=at(1))
    log.append(Actor.USER, EventType.MESSAGE, "b", ts=at(2))
    assert len(log) == 2
