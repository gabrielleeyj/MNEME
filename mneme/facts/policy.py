"""Write-policy seam — the ablation axis of the project.

Three policies share one extractor and one store, so the comparison is
structural and free:
  * ``InsertOnlyPolicy`` — insert every candidate, no conflict handling. The
    workstream-1 default that keeps the rebuild path runnable.
  * ``SupersedePolicy``  — the thesis. Run the contradiction detector; on a
    conflict, close out the old fact and link the new one, keeping full history.
  * ``OverwritePolicy``  — the B0 ablation. Last-write-wins on the
    subject+predicate slot, mutating in place so history is lost. The number the
    whole project is measured against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mneme.domain.events import Event
from mneme.domain.facts import ExtractedFact
from mneme.facts.detector import Detector, Relation

if TYPE_CHECKING:
    from mneme.facts.candidates import CandidateProvider
    from mneme.facts.store import FactStore


class WritePolicy(Protocol):
    def apply(self, store: FactStore, candidate: ExtractedFact, event: Event) -> None:
        """Integrate one extracted fact into the store."""
        ...


class InsertOnlyPolicy:
    """Insert every candidate as a fresh current fact. No conflict handling."""

    def apply(self, store: FactStore, candidate: ExtractedFact, event: Event) -> None:
        store.insert(candidate, event.event_id, ingested_at=event.ts)


class SupersedePolicy:
    """Detect conflicts and resolve them by supersession, preserving history.

    The candidate is judged against the facts the provider surfaces. A DUPLICATE
    is dropped (the belief is already held). Anything else is inserted; a
    REFINES or SUPERSEDES additionally closes out the fact it replaces, on both
    temporal axes: the old fact's valid-time ends where the new one's begins
    (``candidate.valid_from``), and its transaction-time ends now (``event.ts``).
    """

    def __init__(self, detector: Detector, provider: CandidateProvider) -> None:
        self._detector = detector
        self._provider = provider

    def apply(self, store: FactStore, candidate: ExtractedFact, event: Event) -> None:
        existing = self._provider.fetch(candidate)
        judgment = self._detector.judge(candidate, existing)

        if judgment.relation is Relation.DUPLICATE:
            return

        stored = store.insert(candidate, event.event_id, ingested_at=event.ts)
        self._provider.note(stored)

        if (
            judgment.relation in (Relation.SUPERSEDES, Relation.REFINES)
            and judgment.target_fact_id is not None
        ):
            store.close_out(
                judgment.target_fact_id,
                stored.fact_id,
                valid_to=candidate.valid_from,
                superseded_at=event.ts,
            )


class OverwritePolicy:
    """B0 ablation: last-write-wins on the subject+predicate slot, in place.

    No detector, no neighbours — just a dumb exact-slot match. If a current fact
    already exists for this subject+predicate, its value is overwritten and the
    prior value is gone; otherwise the candidate is inserted. This is the weak
    baseline supersession must beat to justify itself.
    """

    def apply(self, store: FactStore, candidate: ExtractedFact, event: Event) -> None:
        existing = store.current_for(candidate.subject, candidate.predicate)
        if existing is None:
            store.insert(candidate, event.event_id, ingested_at=event.ts)
        else:
            store.overwrite(
                existing.fact_id,
                candidate,
                event.event_id,
                ingested_at=event.ts,
            )
