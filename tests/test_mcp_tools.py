"""The MCP tool functions — string formatting over a MemoryService."""

from __future__ import annotations

from mneme.domain.events import Actor
from mneme.mcp import tools
from mneme.service.memory import MemoryService

_MAP = {
    "berlin": [("alice", "lives_in", "Berlin")],
    "lisbon": [("alice", "lives_in", "Lisbon")],
}


def _service(conn, routed_llm):
    return MemoryService(conn, llm=routed_llm(_MAP))


def test_remember_then_recall(conn, routed_llm):
    service = _service(conn, routed_llm)

    assert tools.remember(service, "Alice lives in Berlin") == "Remembered."
    assert tools.recall(service, "alice", "lives_in") == "alice's lives_in is Berlin."


def test_remember_blank_is_noop(conn):
    service = MemoryService(conn)

    assert "Nothing to remember" in tools.remember(service, "   ")


def test_remember_without_key_defers(conn):
    service = MemoryService(conn)  # no llm

    message = tools.remember(service, "Alice lives in Berlin")

    assert "Captured to the log" in message
    assert "remember_fact" in message


def test_remember_fact_stores_without_key(conn):
    service = MemoryService(conn)  # no llm

    message = tools.remember_fact(service, "alice", "lives_in", "Berlin")

    assert message == "Stored: alice's lives_in is Berlin."
    assert tools.recall(service, "alice", "lives_in") == "alice's lives_in is Berlin."


def test_remember_fact_requires_all_components(conn):
    service = MemoryService(conn)

    assert "required" in tools.remember_fact(service, "alice", "lives_in", "  ")


def test_remember_fact_rejects_bad_valid_from(conn):
    service = MemoryService(conn)

    message = tools.remember_fact(service, "alice", "lives_in", "Berlin", "not-a-date")

    assert "Could not parse valid_from" in message


def test_consolidate_without_key_hands_turns_to_agent(conn):
    service = MemoryService(conn)
    service.capture(Actor.USER, "Alice lives in Berlin")

    message = tools.consolidate(service)

    assert "remember_fact" in message
    assert "Alice lives in Berlin" in message


def test_consolidate_without_key_nothing_pending(conn):
    service = MemoryService(conn)

    assert tools.consolidate(service) == "Nothing to consolidate: no pending turns."


def test_recall_missing_slot(conn):
    service = MemoryService(conn)

    assert tools.recall(service, "alice", "lives_in") == (
        "No current belief for alice's lives_in."
    )


def test_evolution_and_history(conn, routed_llm):
    service = _service(conn, routed_llm)
    tools.remember(service, "Alice lives in Berlin")
    tools.remember(service, "Alice now lives in Lisbon")

    assert tools.evolution(service, "alice", "lives_in") == (
        "alice's lives_in over time: Berlin -> Lisbon."
    )
    answer = tools.history(service, "alice", "lives_in", "2030-01-01")
    assert "Lisbon" in answer


def test_history_rejects_bad_date(conn):
    service = MemoryService(conn)

    message = tools.history(service, "alice", "lives_in", "not-a-date")

    assert "Could not parse a date" in message


def test_consolidate_reports_count(conn, routed_llm):
    service = _service(conn, routed_llm)
    service.capture(Actor.USER, "Alice lives in Berlin")

    assert tools.consolidate(service) == "Consolidated 1 event(s) into memory."




def test_memory_summary_lists_facts(conn, routed_llm):
    service = _service(conn, routed_llm)
    tools.remember(service, "Alice lives in Berlin")

    summary = tools.memory_summary(service)

    assert summary.startswith("Known facts from earlier conversations:")
    assert "alice lives_in Berlin" in summary


def test_memory_summary_empty(conn):
    service = MemoryService(conn)

    assert tools.memory_summary(service) == ""
