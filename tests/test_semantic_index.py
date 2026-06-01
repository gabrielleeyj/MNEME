from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from mneme.domain.events import Actor, Event, EventType
from mneme.domain.facts import ExtractedFact
from mneme.embeddings.client import EmbeddingClient
from mneme.index.render import embedding_text_for_event, embedding_text_for_fact
from mneme.index.semantic_index import SemanticIndex


def _fact(subject: str, predicate: str, obj: str) -> ExtractedFact:
    return ExtractedFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_fake_embedder_satisfies_protocol(fake_embedder):
    assert isinstance(fake_embedder, EmbeddingClient)


def test_empty_index_search_returns_empty(fake_embedder):
    index = SemanticIndex(fake_embedder)
    assert index.search("alice lives_in berlin") == []
    assert len(index) == 0


def test_search_returns_most_similar_fact(fake_embedder):
    index = SemanticIndex(fake_embedder)
    index.add(1, embedding_text_for_fact(_fact("alice", "lives_in", "berlin")))
    index.add(2, embedding_text_for_fact(_fact("bob", "likes", "tea")))
    index.add(3, embedding_text_for_fact(_fact("alice", "works_at", "acme")))

    hits = index.search("alice lives_in berlin", k=3)

    assert hits[0][0] == 1  # exact triple match ranks first


def test_search_separates_unrelated_topics(fake_embedder):
    index = SemanticIndex(fake_embedder)
    index.add(1, embedding_text_for_fact(_fact("alice", "lives_in", "berlin")))
    index.add(2, embedding_text_for_fact(_fact("bob", "likes", "tea")))

    assert index.search("tea", k=1)[0][0] == 2


def test_add_many_batches_into_a_single_embed_call(fake_embedder):
    index = SemanticIndex(fake_embedder)
    index.add_many(
        [
            (1, embedding_text_for_fact(_fact("alice", "lives_in", "berlin"))),
            (2, embedding_text_for_fact(_fact("bob", "likes", "tea"))),
        ]
    )
    assert len(index) == 2
    # One add_many -> one batched embed call (plus none for the empty search guard).
    assert len(fake_embedder.calls) == 1
    assert len(fake_embedder.calls[0]) == 2


def test_len_tracks_added_items(fake_embedder):
    index = SemanticIndex(fake_embedder)
    assert len(index) == 0
    index.add(1, "alice lives_in berlin")
    assert len(index) == 1


def test_add_many_empty_is_a_noop(fake_embedder):
    index = SemanticIndex(fake_embedder)
    index.add_many([])
    assert len(index) == 0
    assert fake_embedder.calls == []


def test_event_rendering_indexes_content(fake_embedder):
    index = SemanticIndex(fake_embedder)
    event = Event(
        event_id=7,
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        actor=Actor.USER,
        type=EventType.MESSAGE,
        content="alice lives_in berlin",
    )
    index.add(event.event_id, embedding_text_for_event(event))

    assert index.search("berlin", k=1)[0][0] == 7
