"""SQLite connection management and schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection with row access by name and FK enforcement on.

    For on-disk databases (the plugin runs a long-lived MCP server and short-lived
    hook processes against the same file), WAL mode plus a busy timeout let those
    concurrent writers coexist instead of tripping ``database is locked``. In-memory
    databases are single-connection by nature, so neither pragma applies.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(db_path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create tables, triggers, and indexes if they do not yet exist."""
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()


def init_db(db_path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection and apply the schema. The common entry point."""
    conn = connect(db_path)
    apply_schema(conn)
    return conn
