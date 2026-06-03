"""The projection bookkeeping store — small key/value state over the `meta` table.

Just the consolidation high-water mark for now: the ``event_id`` of the last
event folded into ``facts``. Incremental consolidation reads it to know where to
resume, so a session only pays to extract the turns it has not seen. It is
derived state (a rebuild resets it to zero and replays), never the source of
truth.
"""

from __future__ import annotations

import sqlite3

__all__ = ["get_watermark", "set_watermark", "WATERMARK_KEY"]

WATERMARK_KEY = "consolidated_through_event_id"


def get_watermark(conn: sqlite3.Connection) -> int:
    """The last event_id folded into facts, or 0 if nothing has been consolidated."""
    row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (WATERMARK_KEY,)
    ).fetchone()
    return int(row["value"]) if row is not None else 0


def set_watermark(conn: sqlite3.Connection, event_id: int) -> None:
    """Record that consolidation has folded every event through ``event_id``."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (WATERMARK_KEY, str(event_id)),
    )
    conn.commit()
