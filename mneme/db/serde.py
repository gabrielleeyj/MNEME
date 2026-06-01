"""Boundary serialization helpers: datetime <-> ISO8601 text.

Timestamps live as TEXT (ISO8601) in SQLite and as aware ``datetime`` objects
in Python. Conversion happens only at the storage boundary.
"""

from __future__ import annotations

from datetime import datetime


def to_iso(dt: datetime | None) -> str | None:
    """Serialize a datetime to ISO8601 text, passing NULL through."""
    if dt is None:
        return None
    return dt.isoformat()


def from_iso(text: str | None) -> datetime | None:
    """Parse ISO8601 text to a datetime, passing NULL through."""
    if text is None:
        return None
    return datetime.fromisoformat(text)
