#!/usr/bin/env python3
"""Eyeball semantic retrieval over a handful of facts.

No API key needed — embeddings run locally via fastembed (ONNX). The model is
downloaded once on first run.
Install the extras first:  pip install -e '.[vectors,embeddings]'
Then run:  python scripts/semantic_demo.py

This is the workstream-4 done check: do related queries surface the right
facts (and keep contradiction pairs close together) without garbage?
"""

from __future__ import annotations

from datetime import datetime, timezone

from mneme.domain.facts import ExtractedFact
from mneme.embeddings.client import FastEmbedEmbeddingClient
from mneme.index.render import embedding_text_for_fact
from mneme.index.semantic_index import SemanticIndex

FACTS: list[tuple[str, str, str]] = [
    ("alice", "lives_in", "Berlin"),
    ("alice", "lives_in", "Lisbon"),  # the contradiction pair with the above
    ("alice", "works_at", "Acme"),
    ("bob", "enjoys", "rock climbing"),
    ("bob", "drinks", "green tea"),
    ("mara", "gave_birth_to", "a baby"),
]

QUERIES = [
    "Where does alice live?",
    "What does bob do for fun?",
    "alice's employer",
]


def _fact(subject: str, predicate: str, obj: str) -> ExtractedFact:
    return ExtractedFact(
        subject=subject,
        predicate=predicate,
        object=obj,
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def main() -> None:
    index = SemanticIndex(FastEmbedEmbeddingClient())
    facts = [_fact(*triple) for triple in FACTS]
    index.add_many(
        [(fact_id, embedding_text_for_fact(fact)) for fact_id, fact in enumerate(facts)]
    )
    print(f"indexed {len(index)} facts\n")

    for query in QUERIES:
        print(f"? {query}")
        for fact_id, score in index.search(query, k=3):
            fact = facts[fact_id]
            print(
                f"  {score:.3f}  ({fact.subject}) -[{fact.predicate}]-> ({fact.object})"
            )
        print()


if __name__ == "__main__":
    main()
