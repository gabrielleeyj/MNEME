-- MNEME schema (workstream 1: schema + append-only event log)
--
-- Two tables, two roles:
--   events  -- the source of truth. Pure append. Never updated, never deleted.
--   facts   -- a derived read-model. Rebuildable from events at any time.
--
-- The append-only invariant on `events` is the architecture, so it is enforced
-- at the storage layer via triggers below, not merely by convention.
--
-- `facts` is intentionally NOT locked down. The fact write policy is the thing
-- being ablated in workstream 3: the Supersede policy only ever performs the
-- close-out write (valid_to / superseded_at / superseded_by), while the
-- Overwrite baseline (B0) legitimately mutates `object` in place. Enforcing
-- fact immutability in the schema would forbid B0, so that guarantee lives in
-- the policy layer, not here.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,                 -- ISO8601; when it happened
    actor           TEXT    NOT NULL CHECK (actor IN ('user', 'assistant', 'tool')),
    type            TEXT    NOT NULL CHECK (type IN ('message', 'tool_call', 'artifact', 'reflection')),
    content         TEXT    NOT NULL,
    parent_event_id INTEGER REFERENCES events(event_id)
);

-- Append-only enforcement: events admit INSERT only.
CREATE TRIGGER IF NOT EXISTS events_no_update
BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: UPDATE is forbidden');
END;

CREATE TRIGGER IF NOT EXISTS events_no_delete
BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only: DELETE is forbidden');
END;

CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject         TEXT    NOT NULL,
    predicate       TEXT    NOT NULL,
    object          TEXT    NOT NULL,
    valid_from      TEXT    NOT NULL,                 -- valid time: true in the world from
    valid_to        TEXT,                             -- NULL = still true
    ingested_at     TEXT    NOT NULL,                 -- transaction time: system learned it
    superseded_at   TEXT,                             -- NULL = current belief
    superseded_by   INTEGER REFERENCES facts(fact_id),-- successor fact
    source_event_id INTEGER NOT NULL REFERENCES events(event_id),
    confidence      REAL
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE INDEX IF NOT EXISTS idx_facts_current ON facts(subject) WHERE superseded_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_source  ON facts(source_event_id);
