"""Path resolution for a project's memory database (scope rules)."""

from __future__ import annotations

from pathlib import Path

from mneme.service.paths import (
    DB_ENV_VAR,
    PROJECT_DIR_NAME,
    SCOPE_ENV_VAR,
    resolve_db_path,
)


def test_default_is_per_project(tmp_path: Path):
    # Arrange
    env: dict[str, str] = {}

    # Act
    path = resolve_db_path(tmp_path, env)

    # Assert
    assert path == tmp_path / PROJECT_DIR_NAME / "memory.db"
    assert path.parent.is_dir()


def test_global_scope_points_at_home(tmp_path: Path):
    path = resolve_db_path(tmp_path, {SCOPE_ENV_VAR: "global"})

    assert path == Path.home() / PROJECT_DIR_NAME / "global.db"


def test_explicit_db_overrides_everything(tmp_path: Path):
    explicit = tmp_path / "custom" / "mneme.db"

    path = resolve_db_path(tmp_path, {DB_ENV_VAR: str(explicit), SCOPE_ENV_VAR: "global"})

    assert path == explicit
    assert path.parent.is_dir()


def test_scope_value_is_case_insensitive(tmp_path: Path):
    path = resolve_db_path(tmp_path, {SCOPE_ENV_VAR: "  GLOBAL  "})

    assert path == Path.home() / PROJECT_DIR_NAME / "global.db"


def test_unknown_scope_falls_back_to_project(tmp_path: Path):
    path = resolve_db_path(tmp_path, {SCOPE_ENV_VAR: "team"})

    assert path == tmp_path / PROJECT_DIR_NAME / "memory.db"


def test_create_parent_false_skips_mkdir(tmp_path: Path):
    target = tmp_path / "nope"

    path = resolve_db_path(target, {}, create_parent=False)

    assert path == target / PROJECT_DIR_NAME / "memory.db"
    assert not path.parent.exists()
