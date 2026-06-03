#!/usr/bin/env python
"""Stop hook: capture the assistant's reply, then consolidate on a threshold.

Fires when the agent finishes responding (Claude Code and Codex both raise this
event). Two jobs:

  1. Append the assistant's final text turn to the log, completing the
     user/assistant pair for this exchange.
  2. If enough turns have piled up un-folded *and* a key is available, run a
     consolidation pass so memory stays close to current without paying the LLM
     on every single turn.

Claude Code runs this hook ``async``, so step 2 is free there. Codex has no async
hooks yet, so step 2 blocks the turn end — ``MNEME_CONSOLIDATE_ON_STOP=false``
turns it off and defers consolidation to ``SessionStart``. Everything is
best-effort: any failure exits 0 silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    assistant_reply,
    consolidate_every,
    open_service,
    read_payload,
    stop_consolidation_enabled,
)


def main() -> None:
    payload = read_payload()
    service = open_service(payload)
    if service is None:
        return

    reply = assistant_reply(payload)
    try:
        from mneme.domain.events import Actor

        if reply:
            service.capture(Actor.ASSISTANT, reply)
    except Exception:
        return

    if stop_consolidation_enabled():
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
