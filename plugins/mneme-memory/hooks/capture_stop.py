#!/usr/bin/env python
"""Stop hook: capture Claude's reply, then consolidate on a threshold.

Fires when Claude finishes responding. Two jobs, both in the background (the hook
is marked ``async`` so it never holds up the UI):

  1. Append Claude's final text turn to the log, completing the user/assistant
     pair for this exchange.
  2. If enough turns have piled up un-folded *and* a key is available, run a
     consolidation pass so memory stays close to current without paying the LLM
     on every single turn.

Everything is best-effort: any failure exits 0 silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    consolidate_every,
    last_assistant_text,
    open_service,
    read_payload,
)


def main() -> None:
    payload = read_payload()
    service = open_service(payload)
    if service is None:
        return

    reply = last_assistant_text(payload.get("transcript_path"))
    try:
        from mneme.domain.events import Actor

        if reply:
            service.capture(Actor.ASSISTANT, reply)
    except Exception:
        return

    _maybe_consolidate(service, consolidate_every(payload))


def _maybe_consolidate(service, threshold: int) -> None:
    try:
        if not service.can_consolidate:
            return
        if service.pending_count() < threshold:
            return
        service.consolidate()
    except Exception:
        return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
