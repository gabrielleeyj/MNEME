"""The MCP server adapter registers the memory tools against a service."""

from __future__ import annotations

import asyncio

import pytest

from mneme.service.memory import MemoryService

pytest.importorskip("mcp.server.fastmcp")

from mneme.mcp.server import build_server  # noqa: E402

_EXPECTED_TOOLS = {
    "remember",
    "recall",
    "history",
    "evolution",
    "consolidate",
    "memory_summary",
}


def test_build_server_registers_all_tools(conn):
    server = build_server(MemoryService(conn))

    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == _EXPECTED_TOOLS


def test_build_server_names_the_server(conn):
    server = build_server(MemoryService(conn), name="custom")

    assert server.name == "custom"
