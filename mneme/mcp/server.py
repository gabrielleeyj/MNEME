"""The MCP server — exposes a MemoryService as tools Claude Code can call.

A thin adapter: it registers the functions in ``tools`` with FastMCP and binds
each to a single long-lived ``MemoryService``. ``mcp`` is a lazy import so the
package installs and the test suite runs without it; the tool logic itself is
tested directly against ``tools``.
"""

from __future__ import annotations

from mneme.mcp import tools
from mneme.service.memory import MemoryService

__all__ = ["build_server"]

_INSTRUCTIONS = (
    "MNEME long-term memory for this project. Use `recall` to look up what is "
    "currently believed about someone or something, `history`/`evolution` to see "
    "how a belief changed over time, and `remember` to store a new fact the user "
    "has stated. When no API key is set, MNEME cannot extract facts itself: it "
    "will hand you the conversation turns and ask you to call `remember_fact` "
    "(subject, predicate, object) with the durable facts you extract. Memory is "
    "captured automatically from the conversation; these tools let you query and "
    "add to it deliberately."
)


def build_server(service: MemoryService, *, name: str = "mneme"):
    """Construct a FastMCP server whose tools are bound to ``service``.

    Imported lazily so importing this module never requires the ``mcp`` package.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "the 'mcp' package is required to run the MNEME MCP server; "
            "install with: pip install 'mneme[mcp]'"
        ) from exc

    server = FastMCP(name, instructions=_INSTRUCTIONS)

    @server.tool()
    def remember(text: str) -> str:
        """Store a fact the user has stated, e.g. 'Alice now lives in Lisbon'."""
        return tools.remember(service, text)

    @server.tool()
    def remember_fact(
        subject: str, predicate: str, object: str, valid_from: str = ""
    ) -> str:
        """Store an extracted fact as a triple (subject, predicate, object).

        Use this after extracting durable facts yourself — e.g. when MNEME asks
        you to, or when no API key is set so it cannot extract on its own. Use a
        stable snake_case predicate (lives_in, works_at, prefers). It supersedes
        any prior value for the slot and keeps the old one as history.
        """
        return tools.remember_fact(service, subject, predicate, object, valid_from)

    @server.tool()
    def recall(subject: str, predicate: str) -> str:
        """Look up the current belief for a subject and predicate (e.g. alice, lives_in)."""
        return tools.recall(service, subject, predicate)

    @server.tool()
    def history(subject: str, predicate: str, as_of: str) -> str:
        """What was believed for a slot at an ISO 8601 instant (e.g. 2026-03-01)."""
        return tools.history(service, subject, predicate, as_of)

    @server.tool()
    def evolution(subject: str, predicate: str) -> str:
        """The full ordered history of a belief, oldest first."""
        return tools.evolution(service, subject, predicate)

    @server.tool()
    def consolidate() -> str:
        """Fold any captured-but-unprocessed conversation turns into queryable facts."""
        return tools.consolidate(service)

    @server.tool()
    def memory_summary() -> str:
        """A compact list of everything currently believed in this project's memory."""
        return tools.memory_summary(service)

    return server
