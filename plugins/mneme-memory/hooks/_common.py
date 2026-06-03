"""Shared plumbing for the MNEME hook scripts.

Every hook is a short-lived process that Claude Code launches with the event
payload on stdin. The cardinal rule here is *never break the host session*: a
hook that raises, or writes garbage to stdout, degrades the user's Claude Code.
So everything in this module is defensive — failures are swallowed and the hook
exits 0 with no output. Memory is a nice-to-have; the session is not.

The functions stay dependency-light (only stdlib + ``mneme``) and import lazily
so a machine without MNEME installed simply no-ops instead of crashing.
"""

from __future__ import annotations

import json
import sys
from typing import Any

#: Consolidate once this many un-folded turns have piled up (overridable by env).
DEFAULT_CONSOLIDATE_EVERY = 6
CONSOLIDATE_EVERY_ENV = "MNEME_CONSOLIDATE_EVERY"


def read_payload() -> dict[str, Any]:
    """Parse the hook's stdin JSON, or an empty dict if anything is off."""
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def project_cwd(payload: dict[str, Any]) -> str:
    """The directory whose memory we read/write — the session's working dir."""
    import os

    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return cwd
    return os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def open_service(payload: dict[str, Any]):
    """Open the MemoryService for this session, or ``None`` if unavailable.

    Returns ``None`` (rather than raising) when MNEME is not importable or the
    database cannot be opened, so a hook can always degrade to a silent no-op.
    """
    try:
        import os

        from mneme.service.factory import open_service as _open

        return _open(project_cwd(payload), os.environ)
    except Exception:
        return None


def consolidate_every(payload: dict[str, Any]) -> int:
    """How many pending turns trigger a background consolidation."""
    import os

    raw = os.environ.get(CONSOLIDATE_EVERY_ENV, "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return DEFAULT_CONSOLIDATE_EVERY


def emit_context(event_name: str, context: str) -> None:
    """Print a hook result that injects ``context`` into Claude's view."""
    if not context.strip():
        return
    payload = {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": context,
        }
    }
    print(json.dumps(payload))


def last_assistant_text(transcript_path: str | None) -> str:
    """Pull the most recent assistant *text* turn out of a transcript JSONL.

    Claude Code writes one JSON object per line; assistant turns carry a list of
    content blocks, of which we want only the ``text`` ones (tool calls are not
    durable conversation). Returns the concatenated text of the last assistant
    message that had any, or ``""`` if the transcript is missing or unreadable.
    """
    if not transcript_path:
        return ""
    try:
        with open(transcript_path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return ""

    latest = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(obj, dict) or obj.get("type") != "assistant":
            continue
        text = _message_text(obj.get("message"))
        if text:
            latest = text
    return latest


def _message_text(message: Any) -> str:
    """Join the ``text`` blocks of one assistant message into a single string."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts).strip()
