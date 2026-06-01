from __future__ import annotations

from collections.abc import Sequence

from mneme.domain.events import Actor, Event, EventType
from mneme.domain.facts import ExtractedFact, Fact
from mneme.facts.candidates import SubjectCandidateProvider
from mneme.facts.detector import Judgment, Relation
from mneme.facts.policy import SupersedePolicy


class ScriptedDetector:
    """Decides via an injected rule over the real (candidate, existing) inputs."""

    def __init__(self, decide):
        self._decide = decide
        self.seen: list[tuple[ExtractedFact, list[Fact]]] = []

    def judge(self, candidate: ExtractedFact, existing: Sequence[Fact]) -> Judgment:
        existing = list(existing)
        self.seen.append((candidate, existing))
        return self._decide(candidate, existing)


class RecordingProvider:
    """Wraps a provider and records every fact handed to ``note``."""

    def __init__(self, inner):
        self._inner = inner
        self.noted: list[Fact] = []

    def fetch(self, candidate: ExtractedFact) -> Sequence[Fact]:
        return self._inner.fetch(candidate)

    def note(self, fact: Fact) -> None:
        self.noted.append(fact)
        self._inner.note(fact)


def _supersede_first(candidate, existing):
    if not existing:
        return Judgment(Relation.NEW, None, "first sighting")
    return Judgment(Relation.SUPERSEDES, existing[0].fact_id, "value changed")


def _append(log, content: str, ts) -> Event:
    return log.append(Actor.USER, EventType.MESSAGE, content, ts=ts)


def _fact(subject: str, predicate: str, obj: str, valid_from) -> ExtractedFact:
    return ExtractedFact(subject, predicate, obj, valid_from=valid_from)


def test_first_fact_is_inserted_as_new(conn, log, store, at):
    policy = SupersedePolicy(ScriptedDetector(_supersede_first), SubjectCandidateProvider(conn))
    event = _append(log, "alice in Berlin", at(1))

    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), event)

    current = store.current_facts()
    assert len(current) == 1
    assert current[0].object == "Berlin"
    assert current[0].is_current


def test_supersession_closes_out_old_and_links_successor(conn, log, store, at):
    policy = SupersedePolicy(ScriptedDetector(_supersede_first), SubjectCandidateProvider(conn))

    e1 = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), e1)
    e2 = _append(log, "alice in Lisbon", at(30))
    policy.apply(store, _fact("alice", "lives_in", "Lisbon", at(30)), e2)

    # Both facts persist; only the new one is current.
    assert len(store.all_facts()) == 2
    current = store.current_facts()
    assert [f.object for f in current] == ["Lisbon"]

    old = store.all_facts()[0]
    new = current[0]
    assert not old.is_current
    assert old.superseded_by == new.fact_id
    assert old.valid_to == at(30)  # old belief ended where the new one began
    assert old.superseded_at == at(30)  # learned at the event time


def test_duplicate_is_dropped_without_inserting(conn, log, store, at):
    def decide(candidate, existing):
        if not existing:
            return Judgment(Relation.NEW, None, "first")
        return Judgment(Relation.DUPLICATE, existing[0].fact_id, "same belief")

    policy = SupersedePolicy(ScriptedDetector(decide), SubjectCandidateProvider(conn))

    e1 = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), e1)
    e2 = _append(log, "alice still in Berlin", at(5))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(5)), e2)

    assert len(store.all_facts()) == 1  # the duplicate was not stored


def test_refines_also_closes_out_the_vaguer_fact(conn, log, store, at):
    def decide(candidate, existing):
        if not existing:
            return Judgment(Relation.NEW, None, "first")
        return Judgment(Relation.REFINES, existing[0].fact_id, "more specific")

    policy = SupersedePolicy(ScriptedDetector(decide), SubjectCandidateProvider(conn))

    e1 = _append(log, "alice in Germany", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Germany", at(1)), e1)
    e2 = _append(log, "alice in Berlin", at(2))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(2)), e2)

    current = store.current_facts()
    assert [f.object for f in current] == ["Berlin"]
    assert store.all_facts()[0].superseded_by == current[0].fact_id


def test_note_is_called_on_insert_but_not_on_duplicate(conn, log, store, at):
    def decide(candidate, existing):
        if not existing:
            return Judgment(Relation.NEW, None, "first")
        return Judgment(Relation.DUPLICATE, existing[0].fact_id, "same")

    provider = RecordingProvider(SubjectCandidateProvider(conn))
    policy = SupersedePolicy(ScriptedDetector(decide), provider)

    e1 = _append(log, "alice in Berlin", at(1))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(1)), e1)
    assert len(provider.noted) == 1  # the inserted fact was announced

    e2 = _append(log, "alice in Berlin", at(2))
    policy.apply(store, _fact("alice", "lives_in", "Berlin", at(2)), e2)
    assert len(provider.noted) == 1  # duplicate stored nothing, so nothing noted


def test_supersession_chain_is_walkable(conn, log, store, at):
    policy = SupersedePolicy(ScriptedDetector(_supersede_first), SubjectCandidateProvider(conn))

    for day, city in [(1, "Berlin"), (10, "Lisbon"), (20, "Madrid")]:
        event = _append(log, f"alice in {city}", at(day))
        policy.apply(store, _fact("alice", "lives_in", city, at(day)), event)

    # One current belief, and a chain of three back through superseded_by.
    current = store.current_facts()
    assert [f.object for f in current] == ["Madrid"]

    chain = []
    cursor = current[0]
    while cursor is not None:
        chain.append(cursor.object)
        prev = [f for f in store.all_facts() if f.superseded_by == cursor.fact_id]
        cursor = prev[0] if prev else None
    assert chain == ["Madrid", "Lisbon", "Berlin"]
