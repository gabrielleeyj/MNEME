"""Fact domain models — the derived, rebuildable read-model.

``Fact`` is a persisted row in the facts projection. ``ExtractedFact`` is the
candidate a (future) extractor pulls from an event before it is written; it
carries no identity, validity-close-out, or provenance yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """A fact candidate produced from an event, prior to storage."""

    subject: str
    predicate: str
    object: str
    valid_from: datetime
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Fact:
    """A stored fact. Bitemporal: valid-time and transaction-time axes.

    A fact is *current* when ``superseded_at is None``. The close-out write
    (setting ``valid_to`` / ``superseded_at`` / ``superseded_by``) is the only
    mutation MNEME's Supersede policy performs.
    """

    fact_id: int
    subject: str
    predicate: str
    object: str
    valid_from: datetime
    ingested_at: datetime
    source_event_id: int
    valid_to: datetime | None = None
    superseded_at: datetime | None = None
    superseded_by: int | None = None
    confidence: float | None = None

    @property
    def is_current(self) -> bool:
        """True while this fact is still the system's belief."""
        return self.superseded_at is None
