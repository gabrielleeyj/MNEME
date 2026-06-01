from __future__ import annotations

import pytest

from mneme.domain.events import Actor, EventType
from mneme.domain.facts import ExtractedFact
from mneme.facts.candidates import (
    CandidateProvider,
    SemanticCandidateProvider,
    SubjectCandidateProvider,
)


def _seed_event(log, at):
    return log.append(Actor.USER, EventType.MESSAGE, "seed", ts=at(1))


def _fact(subject, predicate, obj, *, valid_from):
    return ExtractedFact(subject, predicate, obj, valid_from=valid_from)


def _candidate(subject="alice", predicate="lives_in", obj="Lisbon"):
    from datetime import datetime, timezone

    return ExtractedFact(
        subject, predicate, obj, valid_from=datetime(2026, 2, 1, tzinfo=timezone.utc)
    )


def test_subject_provider_returns_only_that_subjects_current_facts(conn, log, store, at):
    event = _seed_event(log, at)
    store.insert(_fact("alice", "lives_in", "Berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))
    store.insert(_fact("alice", "works_at", "Acme", valid_from=at(1)), event.event_id, ingested_at=at(1))
    store.insert(_fact("bob", "lives_in", "Oslo", valid_from=at(1)), event.event_id, ingested_at=at(1))

    provider = SubjectCandidateProvider(conn)
    objects = {f.object for f in provider.fetch(_candidate(subject="alice"))}
    assert objects == {"Berlin", "Acme"}  # both of alice's, none of bob's


def test_subject_provider_excludes_superseded(conn, log, store, at):
    event = _seed_event(log, at)
    old = store.insert(_fact("alice", "lives_in", "Berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))
    new = store.insert(_fact("alice", "lives_in", "Lisbon", valid_from=at(2)), event.event_id, ingested_at=at(2))
    store.close_out(old.fact_id, new.fact_id, valid_to=at(2), superseded_at=at(2))

    provider = SubjectCandidateProvider(conn)
    objects = {f.object for f in provider.fetch(_candidate(subject="alice"))}
    assert objects == {"Lisbon"}


def test_subject_provider_satisfies_protocol(conn):
    assert isinstance(SubjectCandidateProvider(conn), CandidateProvider)


# --- semantic provider (real FAISS, fake embedder) ---------------------------

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from mneme.index.render import embedding_text_for_fact  # noqa: E402
from mneme.index.semantic_index import SemanticIndex  # noqa: E402


def _index_existing(provider, store):
    for fact in store.all_facts():
        provider.note(fact)


def test_semantic_provider_returns_relevant_current_facts(conn, log, store, at, fake_embedder):
    event = _seed_event(log, at)
    store.insert(_fact("alice", "lives_in", "berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))
    store.insert(_fact("bob", "likes", "tea", valid_from=at(1)), event.event_id, ingested_at=at(1))

    provider = SemanticCandidateProvider(SemanticIndex(fake_embedder), store, top_k=5)
    _index_existing(provider, store)

    hits = provider.fetch(_candidate(subject="alice", predicate="lives_in", obj="berlin"))
    assert hits  # the alice/berlin fact is retrievable
    assert hits[0].subject == "alice"


def test_semantic_provider_filters_out_superseded(conn, log, store, at, fake_embedder):
    event = _seed_event(log, at)
    old = store.insert(_fact("alice", "lives_in", "berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))
    new = store.insert(_fact("alice", "lives_in", "lisbon", valid_from=at(2)), event.event_id, ingested_at=at(2))

    provider = SemanticCandidateProvider(SemanticIndex(fake_embedder), store, top_k=5)
    _index_existing(provider, store)  # both indexed while current
    store.close_out(old.fact_id, new.fact_id, valid_to=at(2), superseded_at=at(2))

    objects = {f.object for f in provider.fetch(_candidate(obj="berlin"))}
    assert "berlin" not in objects  # superseded fact is dropped post-hoc
    assert objects == {"lisbon"}


def test_semantic_provider_note_indexes_new_facts(conn, log, store, at, fake_embedder):
    event = _seed_event(log, at)
    provider = SemanticCandidateProvider(SemanticIndex(fake_embedder), store, top_k=5)

    # Nothing indexed yet -> no candidates.
    assert provider.fetch(_candidate(obj="berlin")) == []

    fact = store.insert(_fact("alice", "lives_in", "berlin", valid_from=at(1)), event.event_id, ingested_at=at(1))
    provider.note(fact)
    assert provider.fetch(_candidate(obj="berlin"))  # now retrievable
