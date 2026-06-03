"""The memory tools, as plain functions over a MemoryService.

Kept free of any MCP types so they are unit-testable offline; ``server`` is the
thin layer that registers these with FastMCP. Each returns a short human-readable
string, which is what Claude reads back as the tool result.
"""

from __future__ import annotations

from datetime import datetime, timezone

from mneme.domain.events import Actor
from mneme.service.memory import MemoryService

__all__ = [
    "remember",
    "recall",
    "history",
    "evolution",
    "consolidate",
    "memory_summary",
    "MAX_SUMMARY_FACTS",
]

MAX_SUMMARY_FACTS = 40


def remember(service: MemoryService, text: str) -> str:
    """Capture a statement and fold it into memory immediately if a key is set."""
    event = service.capture(Actor.USER, text)
    if event is None:
        return "Nothing to remember: the text was empty."
    if not service.can_consolidate:
        return (
            "Captured to the log. It will become queryable once consolidation runs "
            "(no ANTHROPIC_API_KEY set, so fact extraction is deferred)."
        )
    service.consolidate()
    return "Remembered."


def recall(service: MemoryService, subject: str, predicate: str) -> str:
    """The current belief for a subject+predicate slot."""
    answer = service.current(subject, predicate)
    if not answer.objects:
        return f"No current belief for {subject}'s {predicate}."
    return f"{subject}'s {predicate} is {_join(answer.objects)}."


def history(service: MemoryService, subject: str, predicate: str, as_of: str) -> str:
    """The belief that was in force for a slot at the ``as_of`` instant (ISO 8601)."""
    instant = _parse_instant(as_of)
    if instant is None:
        return f"Could not parse a date from {as_of!r}; use ISO 8601 (e.g. 2026-03-01)."
    answer = service.historical(subject, predicate, instant)
    if not answer.objects:
        return f"No recorded belief for {subject}'s {predicate} as of {as_of}."
    return f"As of {as_of}, {subject}'s {predicate} was {_join(answer.objects)}."


def evolution(service: MemoryService, subject: str, predicate: str) -> str:
    """The full ordered history of a slot, oldest first."""
    answer = service.evolution(subject, predicate)
    if not answer.objects:
        return f"No recorded history for {subject}'s {predicate}."
    return f"{subject}'s {predicate} over time: {' -> '.join(answer.objects)}."


def consolidate(service: MemoryService) -> str:
    """Fold any captured-but-unprocessed turns into facts."""
    if not service.can_consolidate:
        return "Cannot consolidate: no ANTHROPIC_API_KEY is set."
    processed = service.consolidate()
    return f"Consolidated {processed} event(s) into memory."


def memory_summary(service: MemoryService, *, limit: int = MAX_SUMMARY_FACTS) -> str:
    """A compact list of everything currently believed — the recall-at-a-glance view."""
    facts = service.current_facts()
    if not facts:
        return ""
    lines = [
        f"- {fact.subject} {fact.predicate} {fact.object}" for fact in facts[:limit]
    ]
    more = len(facts) - len(lines)
    if more > 0:
        lines.append(f"- (+{more} more)")
    return "Known facts from earlier conversations:\n" + "\n".join(lines)


def _join(objects: tuple[str, ...]) -> str:
    return ", ".join(objects)


def _parse_instant(text: str) -> datetime | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
