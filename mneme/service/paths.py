"""Where a project's memory database lives.

The Claude Code plugin captures into a SQLite file, and the rule for *which*
file is the user's scope choice, resolved from the environment so hooks and the
MCP server agree without passing paths around:

  * ``MNEME_DB`` — an explicit path. Wins over everything; the escape hatch.
  * ``MNEME_SCOPE=global`` — one shared store at ``~/.mneme/global.db``, so memory
    follows the user across every project.
  * default (per-project) — ``<project>/.mneme/memory.db``, scoped to the working
    directory so projects never bleed into each other.

The directory is created on resolve so the first capture cannot fail on a
missing folder.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = ["DB_ENV_VAR", "SCOPE_ENV_VAR", "PROJECT_DIR_NAME", "resolve_db_path"]

DB_ENV_VAR = "MNEME_DB"
SCOPE_ENV_VAR = "MNEME_SCOPE"
PROJECT_DIR_NAME = ".mneme"
_GLOBAL_SCOPE = "global"


def resolve_db_path(
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    *,
    create_parent: bool = True,
) -> Path:
    """Resolve the memory DB path for ``cwd`` under the scope rules above."""
    environ = os.environ if env is None else env

    explicit = environ.get(DB_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
    elif environ.get(SCOPE_ENV_VAR, "").strip().lower() == _GLOBAL_SCOPE:
        path = Path.home() / PROJECT_DIR_NAME / "global.db"
    else:
        path = Path(cwd).expanduser() / PROJECT_DIR_NAME / "memory.db"

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
