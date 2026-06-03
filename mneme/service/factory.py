"""Open a MemoryService the way both the hooks and the MCP server need it.

One entry point so every process resolves the same database (scope rules in
``paths``) and wires the LLM client the same way: present only when
``ANTHROPIC_API_KEY`` is set, so a keyless environment still captures turns and
simply defers extraction. Building the client never raises here — a missing or
broken client degrades to capture-only, never to a crashed hook.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from mneme.db.connection import init_db
from mneme.llm.client import LLMClient
from mneme.service.memory import MemoryService
from mneme.service.paths import resolve_db_path

__all__ = ["open_service", "build_llm_client"]


def build_llm_client(env: Mapping[str, str] | None = None) -> LLMClient | None:
    """An Anthropic client when a key is present, else ``None`` (capture-only)."""
    environ = os.environ if env is None else env
    api_key = environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        from mneme.llm.client import AnthropicClient

        return AnthropicClient(api_key=api_key)
    except Exception:
        return None


def open_service(
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
) -> MemoryService:
    """Resolve the DB for ``cwd``, apply the schema, and bind a MemoryService."""
    db_path = resolve_db_path(cwd, env)
    conn = init_db(db_path)
    return MemoryService(conn, llm=build_llm_client(env))
