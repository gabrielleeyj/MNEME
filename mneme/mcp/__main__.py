"""Entry point: ``python -m mneme.mcp`` runs the memory server over stdio.

Resolves the project database from the environment (the plugin sets the cwd and
any ``MNEME_*`` scope vars), binds a MemoryService, and serves it. This is the
command the plugin's ``.mcp.json`` launches.
"""

from __future__ import annotations

import os

from mneme.mcp.server import build_server
from mneme.service.factory import open_service


def main() -> None:
    service = open_service(os.getcwd(), os.environ)
    build_server(service).run()


if __name__ == "__main__":
    main()
