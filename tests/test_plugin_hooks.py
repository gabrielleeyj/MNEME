"""The Claude Code plugin hooks: transcript parsing and end-to-end capture.

Unit-tests the pure helpers in ``_common`` and runs the hook scripts as real
subprocesses (the way Claude Code invokes them) to prove they capture turns and
never exit non-zero, even on garbage input.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sqlite3
import sys
from pathlib import Path

import pytest

_HOOKS_DIR = Path(__file__).resolve().parents[1] / "plugins" / "mneme-memory" / "hooks"


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "_mneme_hook_common", _HOOKS_DIR / "_common.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_common()


# --- _common.last_assistant_text -----------------------------------------------


def _write_transcript(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_last_assistant_text_takes_latest_text(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "first"}]}},
            {"type": "user", "message": {"content": "ignored"}},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "second"}]}},
        ],
    )

    assert common.last_assistant_text(str(transcript)) == "second"


def test_last_assistant_text_skips_tool_only_turns(tmp_path: Path):
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "answer"}]}},
            {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "x"}]}},
        ],
    )

    # The trailing tool-only turn has no text, so the prior text turn wins.
    assert common.last_assistant_text(str(transcript)) == "answer"


def test_last_assistant_text_handles_missing_file(tmp_path: Path):
    assert common.last_assistant_text(str(tmp_path / "nope.jsonl")) == ""
    assert common.last_assistant_text("") == ""
    assert common.last_assistant_text(None) == ""


def test_read_payload_tolerates_garbage(monkeypatch):
    monkeypatch.setattr(common.sys, "stdin", _FakeStdin("{not json"))
    assert common.read_payload() == {}


class _FakeStdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


# --- subprocess integration ----------------------------------------------------


def _run_hook(script: str, payload: dict, env_db: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_HOOKS_DIR / script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={"MNEME_DB": str(env_db), "PATH": _path_env()},
    )


def _path_env() -> str:
    import os

    return os.environ.get("PATH", "")


def _events(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return list(
            conn.execute("SELECT actor, type, content FROM events ORDER BY event_id")
        )
    finally:
        conn.close()


def test_capture_prompt_records_user_turn(tmp_path: Path):
    db = tmp_path / "memory.db"
    result = _run_hook(
        "capture_prompt.py",
        {"prompt": "Alice lives in Berlin", "cwd": str(tmp_path)},
        db,
    )

    assert result.returncode == 0
    assert _events(db) == [("user", "message", "Alice lives in Berlin")]


def test_capture_stop_records_assistant_turn(tmp_path: Path):
    db = tmp_path / "memory.db"
    transcript = tmp_path / "t.jsonl"
    _write_transcript(
        transcript,
        [{"type": "assistant", "message": {"content": [{"type": "text", "text": "Noted."}]}}],
    )

    result = _run_hook(
        "capture_stop.py",
        {"cwd": str(tmp_path), "transcript_path": str(transcript)},
        db,
    )

    assert result.returncode == 0
    assert _events(db) == [("assistant", "message", "Noted.")]


def test_session_start_emits_summary_for_known_facts(tmp_path: Path):
    db = tmp_path / "memory.db"
    _seed_fact(db)

    result = _run_hook(
        "session_start.py",
        {"cwd": str(tmp_path), "source": "startup"},
        db,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    context = payload["hookSpecificOutput"]["additionalContext"]
    assert "alice lives_in Lisbon" in context


def test_hooks_survive_empty_stdin(tmp_path: Path):
    db = tmp_path / "memory.db"
    for script in ("capture_prompt.py", "capture_stop.py", "session_start.py"):
        result = subprocess.run(
            [sys.executable, str(_HOOKS_DIR / script)],
            input="",
            capture_output=True,
            text=True,
            env={"MNEME_DB": str(db), "PATH": _path_env()},
        )
        assert result.returncode == 0, script


def _seed_fact(db: Path) -> None:
    from datetime import datetime, timezone

    from mneme.db import init_db
    from mneme.domain.events import Actor, EventType
    from mneme.domain.facts import ExtractedFact
    from mneme.facts.store import FactStore
    from mneme.log.event_log import EventLog

    conn = init_db(str(db))
    event = EventLog(conn).append(Actor.USER, EventType.MESSAGE, "seed")
    FactStore(conn).insert(
        ExtractedFact(
            subject="alice",
            predicate="lives_in",
            object="Lisbon",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            confidence=0.9,
        ),
        source_event_id=event.event_id,
    )
    conn.close()
