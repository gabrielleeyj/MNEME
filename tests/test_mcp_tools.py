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
    assert "consolidation" in message


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


def test_consolidate_without_key(conn):
    service = MemoryService(conn)

    assert tools.consolidate(service) == "Cannot consolidate: no ANTHROPIC_API_KEY is set."


def test_memory_summary_lists_facts(conn, routed_llm):
    service = _service(conn, routed_llm)
    tools.remember(service, "Alice lives in Berlin")

    summary = tools.memory_summary(service)

    assert summary.startswith("Known facts from earlier conversations:")
    assert "alice lives_in Berlin" in summary


def test_memory_summary_empty(conn):
    service = MemoryService(conn)

    assert tools.memory_summary(service) == ""
