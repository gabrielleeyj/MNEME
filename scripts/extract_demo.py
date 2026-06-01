#!/usr/bin/env python3
"""Eyeball the fact extractor on a handful of sample messages.

Requires a real key:  ANTHROPIC_API_KEY=sk-... python scripts/extract_demo.py
Install the LLM extra first:  pip install -e '.[llm]'

This is the workstream-2 done check: do sane facts come out, without garbage?
"""

from __future__ import annotations

from datetime import datetime, timezone

from mneme.domain.events import Actor, Event, EventType
from mneme.facts.llm_extractor import ExtractionError, LLMExtractor
from mneme.llm.client import AnthropicClient

SAMPLES: list[tuple[str, str]] = [
    ("user", "Hey! I just moved from Lisbon to Berlin last month for a new job at Acme."),
    ("user", "Quick q — what time is the standup tomorrow?"),
    ("user", "I used to love hot yoga but honestly I'm all about climbing now."),
    ("user", "My sister Mara just had a baby, so I'm an uncle as of Tuesday."),
    ("user", "We migrated the billing service off Postgres onto ClickHouse in Q1."),
]


def _event(idx: int, actor: str, content: str) -> Event:
    return Event(
        event_id=idx + 1,
        ts=datetime(2026, 1, idx + 1, 12, 0, 0, tzinfo=timezone.utc),
        actor=Actor(actor),
        type=EventType.MESSAGE,
        content=content,
    )


def main() -> None:
    extractor = LLMExtractor(AnthropicClient())
    for idx, (actor, content) in enumerate(SAMPLES):
        event = _event(idx, actor, content)
        print(f"\n[{event.ts.date()}] {actor}: {content}")
        try:
            facts = extractor.extract(event)
        except ExtractionError as exc:
            print(f"  !! extraction error: {exc}")
            continue
        if not facts:
            print("  (no facts)")
        for fact in facts:
            conf = "" if fact.confidence is None else f"  conf={fact.confidence:.2f}"
            print(
                f"  - ({fact.subject}) -[{fact.predicate}]-> ({fact.object})"
                f"  valid_from={fact.valid_from.date()}{conf}"
            )


if __name__ == "__main__":
    main()
