#!/usr/bin/env python3
"""Eyeball the full supersession pipeline on an evolving timeline.

Needs your Anthropic key (extraction + contradiction judgment); embeddings run
locally via fastembed, no extra key:
    ANTHROPIC_API_KEY=sk-... python scripts/supersede_demo.py
Install extras first:  pip install -e '.[llm,vectors,embeddings]'

This is the workstream-3 done check: as the timeline contradicts itself, does
the store close out stale beliefs and keep a walkable history, instead of either
piling up duplicates or silently overwriting the past?
"""

from __future__ import annotations

from datetime import datetime, timezone

from mneme.db import init_db
from mneme.domain.events import Actor, EventType
from mneme.facts.candidates import SemanticCandidateProvider
from mneme.facts.detector import ContradictionDetector, DetectionError
from mneme.facts.llm_extractor import ExtractionError, LLMExtractor
from mneme.facts.policy import SupersedePolicy
from mneme.facts.store import FactStore
from mneme.embeddings.client import FastEmbedEmbeddingClient
from mneme.index.semantic_index import SemanticIndex
from mneme.llm.client import AnthropicClient
from mneme.log.event_log import EventLog

# (day-of-January, message). Ascending so supersession runs in time order.
TIMELINE: list[tuple[int, str]] = [
    (1, "I just moved to Berlin for a new job at Acme."),
    (8, "Honestly I'm all about climbing these days."),
    (40, "Quick update: I relocated from Berlin to Lisbon last week."),
    (55, "I left Acme — started at Globex this month."),
]


def _ts(day: int) -> datetime:
    return datetime(2026, 1, day, 12, 0, 0, tzinfo=timezone.utc)


def _print_state(store: FactStore) -> None:
    current = store.current_facts()
    print("\n=== current beliefs ===")
    for fact in current:
        print(f"  ({fact.subject}) -[{fact.predicate}]-> ({fact.object})")

    superseded = [f for f in store.all_facts() if not f.is_current]
    if superseded:
        print("\n=== superseded history ===")
        for fact in superseded:
            successor = store.get(fact.superseded_by)
            print(
                f"  ({fact.object}) --superseded by--> ({successor.object})"
                f"  [valid {fact.valid_from.date()} .. {fact.valid_to.date()}]"
            )


def main() -> None:
    conn = init_db(":memory:")
    log = EventLog(conn)
    store = FactStore(conn)

    client = AnthropicClient()
    extractor = LLMExtractor(client)
    index = SemanticIndex(FastEmbedEmbeddingClient())
    provider = SemanticCandidateProvider(index, store)
    policy = SupersedePolicy(ContradictionDetector(client), provider)

    for day, content in TIMELINE:
        event = log.append(Actor.USER, EventType.MESSAGE, content, ts=_ts(day))
        print(f"\n[{event.ts.date()}] {content}")
        try:
            candidates = extractor.extract(event)
        except ExtractionError as exc:
            print(f"  !! extraction error: {exc}")
            continue
        for candidate in candidates:
            try:
                policy.apply(store, candidate, event)
                print(
                    f"  + ({candidate.subject}) -[{candidate.predicate}]-> "
                    f"({candidate.object})"
                )
            except DetectionError as exc:
                print(f"  !! detection error: {exc}")

    _print_state(store)


if __name__ == "__main__":
    main()
