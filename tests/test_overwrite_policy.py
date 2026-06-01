from __future__ import annotations

from mneme.domain.events import Actor, Event, EventType
from mneme.domain.facts import ExtractedFact
from mneme.facts.policy import OverwritePolicy


def _append(log, content: str, ts) -> Event:
    return log.append(Actor.USER, EventType.MESSAGE, content, ts=ts)


def _fact(subject: str, predicate: str, obj: str, valid_from) -> ExtractedFact:
    return ExtractedFact(subject, predicate, obj, valid_from=valid_from)


def test_first_fact_for_a_slot_is_inserted(log, store, at):
    policy = OverwritePolicy()
    event = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), event)

    assert len(store.all_facts()) == 1
    assert store.current_facts()[0].object == "Berlin"


def test_same_slot_overwrites_in_place_and_loses_history(log, store, at):
    policy = OverwritePolicy()

    e1 = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), e1)
    first_id = store.current_facts()[0].fact_id

    e2 = _append(log, "alice in Lisbon", at(30))
    policy.apply(store, _fact("alice", "lives_in", "Lisbon", at(30)), e2)

    facts = store.all_facts()
    assert len(facts) == 1  # no second row — the value was mutated in place
    only = facts[0]
    assert only.fact_id == first_id  # same row
    assert only.object == "Lisbon"  # new value
    assert only.valid_from == at(30)  # provenance moved to the new event
    assert only.source_event_id == e2.event_id
    assert only.is_current
    # The Berlin belief is simply gone — nothing records it ever held.
    assert all(f.object != "Berlin" for f in facts)


def test_different_predicate_is_a_separate_slot(log, store, at):
    policy = OverwritePolicy()

    e1 = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), e1)
    e2 = _append(log, "alice at Acme", at(2))
    policy.apply(store, _fact("alice", "works_at", "Acme", at(2)), e2)

    assert {f.object for f in store.current_facts()} == {"Berlin", "Acme"}


def test_different_subject_is_a_separate_slot(log, store, at):
    policy = OverwritePolicy()

    e1 = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), e1)
    e2 = _append(log, "bob in Berlin", at(2))
    policy.apply(store, _fact("bob", "lives_in", "Berlin", at(2)), e2)

    assert len(store.all_facts()) == 2
