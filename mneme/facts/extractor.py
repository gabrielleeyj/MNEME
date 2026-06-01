"""Extractor seam.

Workstream 2 replaces ``NullExtractor`` with the LLM-backed extractor. The
``Extractor`` protocol is the contract the fact store's rebuild depends on, so
it is fixed here in workstream 1; the rebuild path is exercised end-to-end with
a stub or a test double until the real extractor lands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mneme.domain.events import Event
from mneme.domain.facts import ExtractedFact


@runtime_checkable
class Extractor(Protocol):
    def extract(self, event: Event) -> Sequence[ExtractedFact]:
        """Turn one event into zero or more fact candidates."""
        ...


class NullExtractor:
    """Extracts nothing. The default until workstream 2 lands."""

    def extract(self, event: Event) -> Sequence[ExtractedFact]:
        return ()
