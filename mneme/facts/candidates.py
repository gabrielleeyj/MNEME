"""Which existing facts the detector judges a candidate against.

The detector's accuracy is bounded by what it gets to see: if the conflicting
fact is not in the candidate set, no judge can catch the contradiction. This
module owns that retrieval, behind a ``CandidateProvider`` seam so the policy
does not care how candidates are found.

Two implementations:
  * ``SubjectCandidateProvider`` — every current fact about the same subject.
    Deterministic, no model, no index; the honest floor and the test default.
  * ``SemanticCandidateProvider`` — top-k nearest neighbours from the workstream-4
    vector index, filtered to facts that are still current. This is the
    production wiring: it catches conflicts that share meaning but not the exact
    subject string.

``note`` lets a stateful provider learn about a freshly inserted fact (the
semantic one indexes it); for store-backed providers it is a no-op.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mneme.domain.facts import ExtractedFact, Fact
from mneme.facts.store import _COLUMNS, FactStore, _row_to_fact
from mneme.index.render import embedding_text_for_fact
from mneme.index.semantic_index import SemanticIndex

DEFAULT_TOP_K = 10


@runtime_checkable
class CandidateProvider(Protocol):
    def fetch(self, candidate: ExtractedFact) -> Sequence[Fact]:
        """Existing facts the detector should weigh ``candidate`` against."""
        ...

    def note(self, fact: Fact) -> None:
        """Record a newly stored fact so future fetches can return it."""
        ...


class SubjectCandidateProvider:
    """Every current fact about the candidate's subject, straight from the store."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def fetch(self, candidate: ExtractedFact) -> Sequence[Fact]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM facts "
            "WHERE subject = ? AND superseded_at IS NULL ORDER BY fact_id",
            (candidate.subject,),
        )
        return [_row_to_fact(row) for row in rows]

    def note(self, fact: Fact) -> None:
        # Store-backed: the next fetch reads the table directly.
        return None


class SemanticCandidateProvider:
    """Top-k nearest current facts from the workstream-4 vector index."""

    def __init__(
        self,
        index: SemanticIndex,
        store: FactStore,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._index = index
        self._store = store
        self._top_k = top_k

    def fetch(self, candidate: ExtractedFact) -> Sequence[Fact]:
        hits = self._index.search(embedding_text_for_fact(candidate), k=self._top_k)
        facts: list[Fact] = []
        for fact_id, _score in hits:
            try:
                fact = self._store.get(fact_id)
            except KeyError:
                continue  # index may outlive a wiped/rebuilt store
            if fact.is_current:
                facts.append(fact)
        return facts

    def note(self, fact: Fact) -> None:
        self._index.add(fact.fact_id, embedding_text_for_fact(fact))
