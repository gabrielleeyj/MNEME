"""MemoryService — the one object hooks and the MCP server both drive.

It binds MNEME's spine (event log → fact projection → query router) to the
plugin's two-phase, cost-aware capture model:

  * **capture** appends a raw turn to the event log. Free: no LLM, no extraction.
    This is the always-on half — losing a turn is the only unrecoverable failure,
    so capture stays cheap and total.
  * **consolidate** folds the un-consolidated tail of the log into facts under the
    Supersede policy (extract → judge → supersede). This is the LLM-spending half,
    run on a threshold or at session start, and resumable via a watermark so it
    only pays for turns it has not seen.

Reads (current / historical / evolution) go straight through the router over
whatever has been consolidated so far. The split is the event-sourcing thesis in
miniature: the log is the truth, facts are a projection you can fall behind on and
catch up cheaply.
"""

from __future__ import annotations

import sqlite3
import warnings
from datetime import datetime

from mneme.domain.events import Actor, Event, EventType
from mneme.facts.store import FactStore
from mneme.llm.client import LLMClient
from mneme.log.event_log import EventLog
from mneme.query.router import Answer, QueryRouter
from mneme.service.meta import get_watermark, set_watermark
from mneme.service.pipeline import build_extractor, build_supersede_policy

__all__ = ["MemoryService", "ConsolidationWarning"]


class ConsolidationWarning(UserWarning):
    """A single event failed to extract; it is skipped so the pass can finish."""


class MemoryService:
    """Capture, consolidate, and query one project's memory over a connection."""

    def __init__(self, conn: sqlite3.Connection, *, llm: LLMClient | None = None) -> None:
        self._conn = conn
        self._llm = llm
        self._log = EventLog(conn)
        self._store = FactStore(conn)
        self._router = QueryRouter(self._store)

    @property
    def can_consolidate(self) -> bool:
        """True when an LLM client is wired, so consolidation can spend tokens."""
        return self._llm is not None

    # --- capture (free) -------------------------------------------------------

    def capture(
        self,
        actor: Actor | str,
        text: str,
        *,
        ts: datetime | None = None,
    ) -> Event | None:
        """Append one conversation turn to the log. Blank turns are dropped."""
        content = text.strip()
        if not content:
            return None
        return self._log.append(actor, EventType.MESSAGE, content, ts=ts)

    def pending_count(self) -> int:
        """How many captured events have not been consolidated into facts yet."""
        watermark = get_watermark(self._conn)
        row = self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_id > ?", (watermark,)
        ).fetchone()
        return int(row[0])

    # --- consolidate (LLM) ----------------------------------------------------

    def consolidate(self, *, limit: int | None = None) -> int:
        """Fold un-consolidated events into facts; return how many were processed.

        Resumes from the watermark and advances it as it goes, so a crash mid-pass
        loses no ground. A single event that fails extraction is skipped (with a
        ``ConsolidationWarning``) rather than aborting the whole tail. No-ops to 0
        when no LLM client is wired.
        """
        if self._llm is None:
            return 0

        extractor = build_extractor(self._llm)
        policy = build_supersede_policy(self._llm, self._conn)
        watermark = get_watermark(self._conn)

        processed = 0
        for event in self._log.replay():
            if event.event_id <= watermark:
                continue
            if limit is not None and processed >= limit:
                break
            self._consolidate_one(event, extractor, policy)
            set_watermark(self._conn, event.event_id)
            processed += 1
        return processed

    def _consolidate_one(self, event: Event, extractor, policy) -> None:
        try:
            candidates = extractor.extract(event)
            for candidate in candidates:
                policy.apply(self._store, candidate, event)
        except Exception as exc:  # one bad turn must not abort the pass
            warnings.warn(
                f"skipped event {event.event_id}: {exc}",
                ConsolidationWarning,
                stacklevel=2,
            )

    # --- read -----------------------------------------------------------------

    def current(self, subject: str, predicate: str) -> Answer:
        return self._router.current(subject, predicate)

    def historical(self, subject: str, predicate: str, as_of: datetime) -> Answer:
        return self._router.historical(subject, predicate, as_of)

    def evolution(self, subject: str, predicate: str) -> Answer:
        return self._router.evolution(subject, predicate)

    def current_facts(self):
        """Every belief currently in force — the material for a memory summary."""
        return self._store.current_facts()
