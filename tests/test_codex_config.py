"""The Codex integration config registers the MCP server and the real hooks."""

from __future__ import annotations

import tomllib
from pathlib import Path

_PLUGIN = Path(__file__).resolve().parents[1] / "plugins" / "mneme-memory"
_CONFIG = _PLUGIN / "codex" / "config.example.toml"

_HOOK_EVENTS = {"UserPromptSubmit", "Stop", "SessionStart"}
_HOOK_SCRIPTS = {
    "UserPromptSubmit": "capture_prompt.py",
    "Stop": "capture_stop.py",
    "SessionStart": "session_start.py",
}


def _config() -> dict:
    return tomllib.loads(_CONFIG.read_text(encoding="utf-8"))


def test_config_is_valid_toml():
    assert _config()  # parses without raising


def test_mcp_server_runs_the_memory_module():
    server = _config()["mcp_servers"]["mneme"]

    assert server["command"] == "python"
    assert server["args"] == ["-m", "mneme.mcp"]


def test_every_hook_event_is_registered():
    hooks = _config()["hooks"]

    assert set(hooks) == _HOOK_EVENTS


def test_hooks_point_at_scripts_that_exist():
    hooks = _config()["hooks"]

    for event, script in _HOOK_SCRIPTS.items():
        command = hooks[event][0]["hooks"][0]["command"]
        assert hooks[event][0]["hooks"][0]["type"] == "command"
        assert f"hooks/{script}" in command
        assert (_PLUGIN / "hooks" / script).is_file()
