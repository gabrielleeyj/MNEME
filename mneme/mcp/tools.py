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
    "remember_fact",
    "recall",
    "history",
    "evolution",
    "consolidate",
    "memory_summary",
    "extraction_request",
    "MAX_SUMMARY_FACTS",
    "MAX_EXTRACTION_TURNS",
]

MAX_SUMMARY_FACTS = 40
MAX_EXTRACTION_TURNS = 40

#: Handed to the host agent when no key is set: it does the prose-to-triple
#: extraction the LLM would otherwise do, then writes via ``remember_fact``.
EXTRACTION_INSTRUCTION = (
    "No ANTHROPIC_API_KEY is set for MNEME, so fact extraction falls to you. "
    "From the conversation turns below, pull out durable, atomic facts about the "
    "user or their world and store each by calling the `remember_fact` tool "
    "(subject, predicate, object). Use stable snake_case predicates (e.g. "
    "lives_in, works_at, prefers). Skip ephemeral chatter and anything already "
    "known. Turns:"
)


def remember(service: MemoryService, text: str) -> str:
    """Capture a statement and fold it into memory immediately if a key is set."""
    event = service.capture(Actor.USER, text)
    if event is None:
        return "Nothing to remember: the text was empty."
    if not service.can_consolidate:
        return (
            "Captured to the log. No ANTHROPIC_API_KEY is set, so I can't "
            "auto-extract — pull the durable facts out of that statement yourself "
            "and call `remember_fact` (subject, predicate, object) for each so they "
            "become queryable."
        )
    service.consolidate()
    return "Remembered."


def remember_fact(
    service: MemoryService,
    subject: str,
    predicate: str,
    object: str,
    valid_from: str = "",
) -> str:
    """Store an already-extracted triple — the keyless write path, no LLM.

    ``valid_from`` is an optional ISO 8601 instant for when the fact became true;
    omit it to default to now.
    """
    instant = None
    if valid_from.strip():
        instant = _parse_instant(valid_from)
        if instant is None:
            return (
                f"Could not parse valid_from {valid_from!r}; use ISO 8601 "
                "(e.g. 2026-03-01)."
            )
    fact = service.store_fact(subject, predicate, object, valid_from=instant)
    if fact is None:
        return "Nothing stored: subject, predicate, and object are all required."
    return f"Stored: {fact.subject}'s {fact.predicate} is {fact.object}."


def extraction_request(turns: list[tuple[str, str]]) -> str:
    """Frame pending conversation turns as an extraction task for the host agent."""
    if not turns:
        return ""
    lines = [f"- {actor}: {content}" for actor, content in turns[:MAX_EXTRACTION_TURNS]]
    more = len(turns) - len(lines)
    if more > 0:
        lines.append(f"- (+{more} earlier turn(s))")
    return EXTRACTION_INSTRUCTION + "\n" + "\n".join(lines)


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
    """Fold any captured-but-unprocessed turns into facts.

    With a key, runs the LLM consolidation. Without one, there is no model to
    extract with, so it hands the pending turns back as an extraction task for
    the host agent to complete via ``remember_fact``.
    """
    if service.can_consolidate:
        processed = service.consolidate()
        return f"Consolidated {processed} event(s) into memory."
    turns = service.pending_messages()
    if not turns:
        return "Nothing to consolidate: no pending turns."
    return extraction_request(turns)


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
