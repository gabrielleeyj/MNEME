"""The append-only event log — MNEME's source of truth.

Writes are append-only by construction: this class exposes ``append`` but no
update or delete. The storage layer enforces the same invariant via triggers,
so a stray UPDATE/DELETE aborts even if it bypasses this API.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime, timezone

from mneme.db.serde import from_iso, to_iso
from mneme.domain.events import Actor, Event, EventType


def _row_to_event(row: sqlite3.Row) -> Event:
    ts = from_iso(row["ts"])
    assert ts is not None  # ts is NOT NULL in the schema
    return Event(
        event_id=row["event_id"],
        ts=ts,
        actor=Actor(row["actor"]),
        type=EventType(row["type"]),
        content=row["content"],
        parent_event_id=row["parent_event_id"],
    )


class EventLog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append(
        self,
        actor: Actor | str,
        type: EventType | str,
        content: str,
        *,
        ts: datetime | None = None,
        parent_event_id: int | None = None,
    ) -> Event:
        """Append one event and return it. The only write this log permits.

        Validates the boundary: ``actor``/``type`` must be valid enum members,
        ``content`` must be non-empty, and ``parent_event_id`` (if given) must
        reference an existing event (enforced by the FK).
        """
        actor = Actor(actor)
        type = EventType(type)
        if not content:
            raise ValueError("event content must be non-empty")
        when = ts if ts is not None else datetime.now(timezone.utc)

        cursor = self._conn.execute(
            "INSERT INTO events (ts, actor, type, content, parent_event_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (to_iso(when), actor.value, type.value, content, parent_event_id),
        )
        self._conn.commit()
        return self.get(cursor.lastrowid)

    def get(self, event_id: int) -> Event:
        row = self._conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no event with event_id={event_id}")
        return _row_to_event(row)

    def replay(self) -> Iterator[Event]:
        """Yield every event in append (event_id) order."""
        for row in self._conn.execute("SELECT * FROM events ORDER BY event_id"):
            yield _row_to_event(row)

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
