#!/usr/bin/env python
"""SessionStart hook: catch up on consolidation, then inject what we remember.

Fires when a Claude Code session begins. Two jobs:

  1. Fold any turns captured-but-not-consolidated at the end of the last session
     into facts, so memory reflects everything said so far (best-effort, only
     when a key is present).
  2. Hand Claude a compact summary of the current beliefs as ``additionalContext``
     so it walks into the conversation already knowing what earlier sessions
     established — recall without the model having to ask.

Failures are silent: a session must start whether or not memory is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import emit_context, open_service, read_payload  # noqa: E402

EVENT_NAME = "SessionStart"


def main() -> None:
    payload = read_payload()
    service = open_service(payload)
    if service is None:
        return

    _catch_up(service)

    summary = _summary(service)
    if summary:
        emit_context(EVENT_NAME, summary)


def _catch_up(service) -> None:
    try:
        if service.can_consolidate and service.pending_count() > 0:
            service.consolidate()
    except Exception:
        return


def _summary(service) -> str:
    try:
        from mneme.mcp.tools import memory_summary

        return memory_summary(service)
    except Exception:
        return ""


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
