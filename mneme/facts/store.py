"""The facts read-model — a derived, rebuildable projection of the log.

Two write primitives:
  * ``insert``    — add a new current fact.
  * ``close_out`` — the supersession write (valid_to / superseded_at /
                    superseded_by). The only mutation MNEME's Supersede policy
                    performs on an existing row.

And the property that makes the architecture hold: ``rebuild`` discards the
projection and re-derives it from the event log, proving facts are reconstructible
and that the log — not this table — is the source of truth.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from mneme.db.serde import from_iso, to_iso
from mneme.domain.facts import ExtractedFact, Fact
from mneme.facts.extractor import Extractor
from mneme.facts.policy import InsertOnlyPolicy, WritePolicy
from mneme.log.event_log import EventLog

_COLUMNS = (
    "fact_id, subject, predicate, object, valid_from, valid_to, "
    "ingested_at, superseded_at, superseded_by, source_event_id, confidence"
)


def _row_to_fact(row: sqlite3.Row) -> Fact:
    valid_from = from_iso(row["valid_from"])
    ingested_at = from_iso(row["ingested_at"])
    assert valid_from is not None and ingested_at is not None  # NOT NULL columns
    return Fact(
        fact_id=row["fact_id"],
        subject=row["subject"],
        predicate=row["predicate"],
        object=row["object"],
        valid_from=valid_from,
        ingested_at=ingested_at,
        source_event_id=row["source_event_id"],
        valid_to=from_iso(row["valid_to"]),
        superseded_at=from_iso(row["superseded_at"]),
        superseded_by=row["superseded_by"],
        confidence=row["confidence"],
    )


class FactStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(
        self,
        candidate: ExtractedFact,
        source_event_id: int,
        *,
        ingested_at: datetime | None = None,
    ) -> Fact:
        """Insert a fresh current fact derived from ``source_event_id``."""
        if not candidate.subject or not candidate.predicate or not candidate.object:
            raise ValueError("fact subject, predicate, and object must be non-empty")
        learned = ingested_at if ingested_at is not None else datetime.now(timezone.utc)

        cursor = self._conn.execute(
            "INSERT INTO facts "
            "(subject, predicate, object, valid_from, valid_to, ingested_at, "
            " superseded_at, superseded_by, source_event_id, confidence) "
            "VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?)",
            (
                candidate.subject,
                candidate.predicate,
                candidate.object,
                to_iso(candidate.valid_from),
                to_iso(learned),
                source_event_id,
                candidate.confidence,
            ),
        )
        self._conn.commit()
        return self.get(cursor.lastrowid)

    def close_out(
        self,
        fact_id: int,
        successor_id: int,
        *,
        valid_to: datetime | None = None,
        superseded_at: datetime | None = None,
    ) -> Fact:
        """Supersede ``fact_id`` with ``successor_id``.

        The single write-after-insert MNEME's Supersede policy is allowed to
        make. ``valid_to`` defaults to the moment of supersession.
        """
        when = superseded_at if superseded_at is not None else datetime.now(timezone.utc)
        ended = valid_to if valid_to is not None else when

        target = self.get(fact_id)  # raises KeyError if absent
        if not target.is_current:
            raise ValueError(f"fact {fact_id} is already superseded")

        self._conn.execute(
            "UPDATE facts SET valid_to = ?, superseded_at = ?, superseded_by = ? "
            "WHERE fact_id = ?",
            (to_iso(ended), to_iso(when), successor_id, fact_id),
        )
        self._conn.commit()
        return self.get(fact_id)

    def current_for(self, subject: str, predicate: str) -> Fact | None:
        """The current fact for a subject+predicate slot, or None.

        Used by the B0 overwrite ablation to find the single belief it will
        replace in place. Supersession keeps at most one current fact per slot,
        so this returns the earliest-inserted current match for safety.
        """
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM facts "
            "WHERE subject = ? AND predicate = ? AND superseded_at IS NULL "
            "ORDER BY fact_id LIMIT 1",
            (subject, predicate),
        ).fetchone()
        return _row_to_fact(row) if row is not None else None

    def overwrite(
        self,
        fact_id: int,
        candidate: ExtractedFact,
        source_event_id: int,
        *,
        ingested_at: datetime | None = None,
    ) -> Fact:
        """Replace a fact's value in place — the B0 ablation's destructive write.

        Last-write-wins: the object, validity, provenance, and confidence are
        overwritten and the prior value is gone. This is the deliberate opposite
        of ``close_out``; it exists only so the overwrite baseline can lose
        history, which is exactly what the thesis measures against.
        """
        if not candidate.subject or not candidate.predicate or not candidate.object:
            raise ValueError("fact subject, predicate, and object must be non-empty")
        learned = ingested_at if ingested_at is not None else datetime.now(timezone.utc)

        self.get(fact_id)  # raises KeyError if absent
        self._conn.execute(
            "UPDATE facts SET subject = ?, predicate = ?, object = ?, "
            "valid_from = ?, ingested_at = ?, source_event_id = ?, confidence = ? "
            "WHERE fact_id = ?",
            (
                candidate.subject,
                candidate.predicate,
                candidate.object,
                to_iso(candidate.valid_from),
                to_iso(learned),
                source_event_id,
                candidate.confidence,
                fact_id,
            ),
        )
        self._conn.commit()
        return self.get(fact_id)

    def get(self, fact_id: int) -> Fact:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no fact with fact_id={fact_id}")
        return _row_to_fact(row)

    def current_facts(self) -> list[Fact]:
        """Every fact that is still believed (not superseded)."""
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM facts WHERE superseded_at IS NULL "
            "ORDER BY fact_id"
        )
        return [_row_to_fact(row) for row in rows]

    def all_facts(self) -> list[Fact]:
        """Every fact, current and superseded, in insertion order."""
        rows = self._conn.execute(f"SELECT {_COLUMNS} FROM facts ORDER BY fact_id")
        return [_row_to_fact(row) for row in rows]

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def rebuild(
        self,
        event_log: EventLog,
        extractor: Extractor,
        policy: WritePolicy | None = None,
    ) -> int:
        """Discard the projection and re-derive it from the log.

        Returns the number of facts in the rebuilt projection. This is the
        operation that proves facts are a pure function of the event log: if the
        table is corrupted or the extraction logic improves, the read-model is
        regenerated from the source of truth.
        """
        active_policy = policy if policy is not None else InsertOnlyPolicy()
        self._wipe()
        for event in event_log.replay():
            for candidate in extractor.extract(event):
                active_policy.apply(self, candidate, event)
        return len(self)

    def _wipe(self) -> None:
        # Drop the self-referential successor links first so the delete cannot
        # trip the facts.superseded_by foreign key, then clear the table and
        # reset the autoincrement counter for a clean re-derivation.
        self._conn.execute("UPDATE facts SET superseded_by = NULL")
        self._conn.execute("DELETE FROM facts")
        self._conn.execute("DELETE FROM sqlite_sequence WHERE name = 'facts'")
        self._conn.commit()
