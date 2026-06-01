"""Write-policy seam — the ablation axis of the project.

Workstream 3 adds the contradiction detector and two real policies:
  * Supersede  — close out the conflicting fact, insert the new one (history kept)
  * Overwrite  — last-write-wins, mutate the object in place (history gone)

Both share one extractor and one store, so the ablation is structural and free.
Workstream 1 ships only the trivial ``InsertOnlyPolicy`` so the rebuild path is
end-to-end runnable; it performs no conflict detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mneme.domain.events import Event
from mneme.domain.facts import ExtractedFact

if TYPE_CHECKING:
    from mneme.facts.store import FactStore


class WritePolicy(Protocol):
    def apply(self, store: FactStore, candidate: ExtractedFact, event: Event) -> None:
        """Integrate one extracted fact into the store."""
        ...


class InsertOnlyPolicy:
    """Insert every candidate as a fresh current fact. No conflict handling."""

    def apply(self, store: FactStore, candidate: ExtractedFact, event: Event) -> None:
        store.insert(candidate, event.event_id, ingested_at=event.ts)
