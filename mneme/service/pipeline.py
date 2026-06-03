"""Assemble the live ingest pipeline the memory service consolidates with.

This is the same Supersede wiring the thesis is built on — `LLMExtractor` →
`ContradictionDetector` → `SupersedePolicy` over a `SubjectCandidateProvider` —
packaged as one factory so the service and its tests build it the same way. The
``LLMClient`` is injected: real ``AnthropicClient`` in production, a fake in
tests, so consolidation is exercised offline.
"""

from __future__ import annotations

import sqlite3

from mneme.facts.candidates import SubjectCandidateProvider
from mneme.facts.detector import ContradictionDetector
from mneme.facts.extractor import Extractor
from mneme.facts.llm_extractor import LLMExtractor
from mneme.facts.policy import SupersedePolicy
from mneme.llm.client import LLMClient

__all__ = ["build_extractor", "build_supersede_policy"]


def build_extractor(client: LLMClient) -> Extractor:
    """The fact extractor used for consolidation."""
    return LLMExtractor(client)


def build_supersede_policy(
    client: LLMClient, conn: sqlite3.Connection
) -> SupersedePolicy:
    """A Supersede policy that judges candidates against same-subject current facts.

    The subject-match provider is the deterministic floor (no vector index needed),
    which keeps the plugin's dependency surface small; semantic candidates remain a
    later upgrade behind the same seam.
    """
    detector = ContradictionDetector(client)
    provider = SubjectCandidateProvider(conn)
    return SupersedePolicy(detector, provider)
