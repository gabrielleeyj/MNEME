"""The MNEME MCP server (workstream 8) — memory tools for Claude Code.

Exposes a project's MemoryService as MCP tools (recall / history / evolution /
remember / consolidate / memory_summary). The tool logic lives in ``tools`` (MCP-
free, unit-testable); ``server`` registers it with FastMCP and ``__main__`` serves
it over stdio.
"""

from mneme.mcp.server import build_server

__all__ = ["build_server"]
