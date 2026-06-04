#!/usr/bin/env python
"""SessionStart hook: catch up on consolidation, then inject what we remember.

Fires when a session begins. It hands the agent a compact summary of the current
beliefs as ``additionalContext`` so it walks in already knowing what earlier
sessions established — recall without the model having to ask. How the pending
tail gets folded in depends on whether a key is set:

  * **With a key**, MNEME consolidates the un-folded turns itself (LLM extract +
    supersede) before building the summary.
  * **Without a key**, there is no model to extract with, so the host agent *is*
    the extractor: the pending turns are handed back as a task to call
    ``remember_fact`` on, and the watermark advances so they are not re-surfaced
    every session.

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

    if service.can_consolidate:
        _catch_up(service)
        context = _summary(service)
    else:
        context = _keyless_context(service)

    if context:
        emit_context(EVENT_NAME, context)


def _catch_up(service) -> None:
    try:
        if service.pending_count() > 0:
            service.consolidate()
    except Exception:
        return


def _keyless_context(service) -> str:
    """Known facts plus, if turns are pending, an extraction task for the agent."""
    try:
        from mneme.mcp.tools import extraction_request, memory_summary

        parts = []
        summary = memory_summary(service)
        if summary:
            parts.append(summary)

        turns = service.pending_messages()
        if turns:
            parts.append(extraction_request(turns))
            service.mark_pending_consolidated()
        return "\n\n".join(parts)
    except Exception:
        return ""


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
