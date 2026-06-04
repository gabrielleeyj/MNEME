"""The MemoryService: capture (free), consolidate (LLM), and read-through."""

from __future__ import annotations

import pytest

from mneme.domain.events import Actor
from mneme.service.memory import MemoryService
from mneme.service.meta import get_watermark

_MAP = {
    "berlin": [("alice", "lives_in", "Berlin")],
    "lisbon": [("alice", "lives_in", "Lisbon")],
}


def test_capture_appends_and_drops_blank(conn):
    # Arrange
    service = MemoryService(conn)

    # Act
    kept = service.capture(Actor.USER, "Alice lives in Berlin")
    blank = service.capture(Actor.USER, "   ")

    # Assert
    assert kept is not None
    assert blank is None
    assert service.pending_count() == 1


def test_no_key_degrades_to_capture_only(conn):
    service = MemoryService(conn)  # no llm

    service.capture(Actor.USER, "Alice lives in Berlin")

    assert service.can_consolidate is False
    assert service.consolidate() == 0
    assert service.current("alice", "lives_in").objects == ()


def test_consolidate_extracts_current_belief(conn, routed_llm):
    service = MemoryService(conn, llm=routed_llm(_MAP))

    service.capture(Actor.USER, "Alice lives in Berlin")
    processed = service.consolidate()

    assert processed == 1
    assert service.current("alice", "lives_in").objects == ("Berlin",)


def test_supersession_keeps_history(conn, routed_llm):
    service = MemoryService(conn, llm=routed_llm(_MAP))

    service.capture(Actor.USER, "Alice lives in Berlin")
    service.capture(Actor.USER, "Alice now lives in Lisbon")
    service.consolidate()

    assert service.current("alice", "lives_in").objects == ("Lisbon",)
    assert service.evolution("alice", "lives_in").objects == ("Berlin", "Lisbon")


def test_watermark_makes_consolidation_incremental(conn, routed_llm):
    service = MemoryService(conn, llm=routed_llm(_MAP))

    service.capture(Actor.USER, "Alice lives in Berlin")
    assert service.consolidate() == 1
    first_mark = get_watermark(conn)

    # Nothing new captured -> nothing reprocessed.
    assert service.consolidate() == 0
    assert get_watermark(conn) == first_mark

    service.capture(Actor.USER, "Alice now lives in Lisbon")
    assert service.pending_count() == 1
    assert service.consolidate() == 1
    assert get_watermark(conn) > first_mark


def test_consolidate_limit_processes_a_prefix(conn, routed_llm):
    service = MemoryService(conn, llm=routed_llm(_MAP))

    service.capture(Actor.USER, "Alice lives in Berlin")
    service.capture(Actor.USER, "Alice now lives in Lisbon")

    assert service.consolidate(limit=1) == 1
    assert service.pending_count() == 1
    assert service.current("alice", "lives_in").objects == ("Berlin",)


# --- store_fact (keyless, deterministic) ---------------------------------------


def test_store_fact_works_without_a_key(conn):
    service = MemoryService(conn)  # no llm

    fact = service.store_fact("alice", "lives_in", "Berlin")

    assert fact is not None
    assert service.current("alice", "lives_in").objects == ("Berlin",)


def test_store_fact_supersedes_and_keeps_history(conn):
    service = MemoryService(conn)

    service.store_fact("alice", "lives_in", "Berlin")
    service.store_fact("alice", "lives_in", "Lisbon")

    assert service.current("alice", "lives_in").objects == ("Lisbon",)
    assert service.evolution("alice", "lives_in").objects == ("Berlin", "Lisbon")


def test_store_fact_drops_duplicate(conn):
    service = MemoryService(conn)

    service.store_fact("alice", "lives_in", "Berlin")
    service.store_fact("alice", "lives_in", "Berlin")

    assert service.evolution("alice", "lives_in").objects == ("Berlin",)


def test_store_fact_rejects_blank_components(conn):
    service = MemoryService(conn)

    assert service.store_fact("  ", "lives_in", "Berlin") is None
    assert service.store_fact("alice", "", "Berlin") is None
    assert service.store_fact("alice", "lives_in", "   ") is None


def test_pending_messages_excludes_agent_reflections(conn):
    service = MemoryService(conn)
    service.capture(Actor.USER, "Alice lives in Berlin")
    service.store_fact("alice", "lives_in", "Berlin")  # appends a reflection event

    # Only the user message is offered to the agent for extraction.
    assert service.pending_messages() == [("user", "Alice lives in Berlin")]


def test_mark_pending_consolidated_advances_watermark(conn):
    service = MemoryService(conn)
    service.capture(Actor.USER, "Alice lives in Berlin")
    assert service.pending_count() == 1

    service.mark_pending_consolidated()

    assert service.pending_count() == 0
    assert service.pending_messages() == []


def test_bad_extraction_is_skipped_not_fatal(conn):
    class Boom:
        def complete(self, *, system, user, max_tokens=None):
            return "not json"

    service = MemoryService(conn, llm=Boom())
    service.capture(Actor.USER, "Alice lives in Berlin")

    with pytest.warns(UserWarning):
        processed = service.consolidate()

    # The pass still advances past the bad turn.
    assert processed == 1
    assert service.pending_count() == 0
