#!/usr/bin/env python
"""UserPromptSubmit hook: capture the user's turn into memory.

Fires the instant the user hits enter, before Claude responds. We append the raw
prompt to the event log — the free, always-on half of the two-phase model — and
get out of the way. No LLM, no extraction, no output: capturing must never add
latency to the user's prompt or risk corrupting the turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import open_service, read_payload  # noqa: E402


def main() -> None:
    payload = read_payload()
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return

    service = open_service(payload)
    if service is None:
        return

    try:
        from mneme.domain.events import Actor

        service.capture(Actor.USER, prompt)
    except Exception:
        return


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
    sys.exit(0)
