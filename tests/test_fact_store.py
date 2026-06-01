from __future__ import annotations

import pytest

from mneme.domain.events import Actor, EventType
from mneme.domain.facts import ExtractedFact


def _candidate(subject="alice", obj="London", *, valid_from, confidence=0.8):
    return ExtractedFact(
        subject=subject,
        predicate="lives_in",
        object=obj,
        valid_from=valid_from,
        confidence=confidence,
    )


def test_insert_creates_current_fact(log, store, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    fact = store.insert(_candidate(valid_from=at(1)), event.event_id, ingested_at=at(1))

    assert fact.fact_id == 1
    assert fact.is_current
    assert fact.valid_to is None
    assert fact.superseded_by is None
    assert fact.source_event_id == event.event_id


def test_insert_rejects_empty_triple_component(log, store, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "x:y", ts=at(1))
    with pytest.raises(ValueError):
        store.insert(_candidate(subject="", valid_from=at(1)), event.event_id)


def test_close_out_supersedes_old_fact(log, store, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    old = store.insert(_candidate(obj="London", valid_from=at(1)), event.event_id)
    new = store.insert(_candidate(obj="Berlin", valid_from=at(10)), event.event_id)

    closed = store.close_out(old.fact_id, new.fact_id, valid_to=at(10), superseded_at=at(10))

    assert not closed.is_current
    assert closed.valid_to == at(10)
    assert closed.superseded_at == at(10)
    assert closed.superseded_by == new.fact_id


def test_current_facts_excludes_superseded(log, store, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    old = store.insert(_candidate(obj="London", valid_from=at(1)), event.event_id)
    new = store.insert(_candidate(obj="Berlin", valid_from=at(10)), event.event_id)
    store.close_out(old.fact_id, new.fact_id)

    current = store.current_facts()
    assert [f.fact_id for f in current] == [new.fact_id]
    assert len(store.all_facts()) == 2


def test_close_out_twice_is_rejected(log, store, at):
    event = log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    old = store.insert(_candidate(obj="London", valid_from=at(1)), event.event_id)
    new = store.insert(_candidate(obj="Berlin", valid_from=at(10)), event.event_id)
    store.close_out(old.fact_id, new.fact_id)
    with pytest.raises(ValueError):
        store.close_out(old.fact_id, new.fact_id)


def test_rebuild_from_empty_log_yields_no_facts(log, store, extractor):
    assert store.rebuild(log, extractor) == 0
    assert store.current_facts() == []


def test_rebuild_derives_facts_from_log(log, store, extractor, at):
    log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    log.append(Actor.USER, EventType.MESSAGE, "bob:Paris", ts=at(2))
    log.append(Actor.ASSISTANT, EventType.REFLECTION, "no fact here", ts=at(3))

    count = store.rebuild(log, extractor)

    assert count == 2
    facts = {(f.subject, f.object) for f in store.current_facts()}
    assert facts == {("alice", "London"), ("bob", "Paris")}


def test_rebuild_is_deterministic_and_repeatable(log, store, extractor, at):
    log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    log.append(Actor.USER, EventType.MESSAGE, "bob:Paris", ts=at(2))

    first = store.rebuild(log, extractor)
    second = store.rebuild(log, extractor)

    assert first == second == 2
    # Identity counter is reset each rebuild, so ids are stable too.
    assert [f.fact_id for f in store.all_facts()] == [1, 2]


def test_rebuild_does_not_touch_the_event_log(log, store, extractor, at):
    log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    log.append(Actor.USER, EventType.MESSAGE, "bob:Paris", ts=at(2))
    before = [e.content for e in log.replay()]

    store.rebuild(log, extractor)

    after = [e.content for e in log.replay()]
    assert before == after
    assert len(log) == 2


def test_rebuild_recovers_a_corrupted_projection(log, store, extractor, conn, at):
    log.append(Actor.USER, EventType.MESSAGE, "alice:London", ts=at(1))
    store.rebuild(log, extractor)

    # Simulate corruption: facts is a derived table, so direct damage is allowed.
    conn.execute("DELETE FROM facts")
    conn.commit()
    assert store.current_facts() == []

    store.rebuild(log, extractor)
    assert {f.subject for f in store.current_facts()} == {"alice"}
