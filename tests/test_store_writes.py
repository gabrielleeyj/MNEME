from __future__ import annotations

import pytest

from mneme.domain.events import Actor, EventType
from mneme.domain.facts import ExtractedFact


def _seed_event(log, at):
    return log.append(Actor.USER, EventType.MESSAGE, "seed", ts=at(1))


def _fact(subject="alice", predicate="lives_in", obj="Berlin", *, valid_from):
    return ExtractedFact(subject, predicate, obj, valid_from=valid_from)


def test_current_for_returns_none_when_empty(store):
    assert store.current_for("alice", "lives_in") is None


def test_current_for_returns_the_current_fact(log, store, at):
    event = _seed_event(log, at)
    store.insert(_fact(valid_from=at(1)), event.event_id, ingested_at=at(1))

    found = store.current_for("alice", "lives_in")
    assert found is not None
    assert found.object == "Berlin"


def test_current_for_ignores_superseded_facts(log, store, at):
    event = _seed_event(log, at)
    old = store.insert(_fact(obj="Berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))
    new = store.insert(_fact(obj="Lisbon", valid_from=at(2)), event.event_id, ingested_at=at(2))
    store.close_out(old.fact_id, new.fact_id, valid_to=at(2), superseded_at=at(2))

    found = store.current_for("alice", "lives_in")
    assert found is not None
    assert found.object == "Lisbon"


def test_overwrite_mutates_in_place_keeping_fact_id(log, store, at):
    event = _seed_event(log, at)
    original = store.insert(_fact(obj="Berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))

    updated = store.overwrite(
        original.fact_id,
        _fact(obj="Lisbon", valid_from=at(30)),
        event.event_id,
        ingested_at=at(30),
    )

    assert updated.fact_id == original.fact_id
    assert updated.object == "Lisbon"
    assert updated.valid_from == at(30)
    assert len(store.all_facts()) == 1


def test_overwrite_unknown_fact_raises(log, store, at):
    _seed_event(log, at)
    with pytest.raises(KeyError):
        store.overwrite(999, _fact(valid_from=at(1)), 1, ingested_at=at(1))


def test_overwrite_rejects_empty_triple(log, store, at):
    event = _seed_event(log, at)
    original = store.insert(_fact(valid_from=at(1)), event.event_id, ingested_at=at(1))
    with pytest.raises(ValueError):
        store.overwrite(
            original.fact_id,
            ExtractedFact("alice", "lives_in", "", valid_from=at(2)),
            event.event_id,
            ingested_at=at(2),
        )
