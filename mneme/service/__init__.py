"""The memory service layer (workstream 8) — MNEME as a usable memory.

Binds the event-sourced spine to a cost-aware capture/consolidate/recall service
that the Claude Code plugin (hooks + MCP server) drives, plus the scope rules for
where a project's database lives.
"""

from mneme.service.factory import build_llm_client, open_service
from mneme.service.memory import ConsolidationWarning, MemoryService
from mneme.service.paths import (
    DB_ENV_VAR,
    PROJECT_DIR_NAME,
    SCOPE_ENV_VAR,
    resolve_db_path,
)

__all__ = [
    "MemoryService",
    "ConsolidationWarning",
    "open_service",
    "build_llm_client",
    "resolve_db_path",
    "DB_ENV_VAR",
    "SCOPE_ENV_VAR",
    "PROJECT_DIR_NAME",
]
