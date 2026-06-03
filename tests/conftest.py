from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from mneme.db import init_db
from mneme.domain.events import Event
from mneme.domain.facts import ExtractedFact
from mneme.facts.store import FactStore
from mneme.log.event_log import EventLog


@pytest.fixture
def conn():
    connection = init_db(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def log(conn):
    return EventLog(conn)


@pytest.fixture
def store(conn):
    return FactStore(conn)


@pytest.fixture
def at():
    """Deterministic timestamp factory: at(day) -> 2026-01-<day> UTC."""

    def _at(day: int, hour: int = 12) -> datetime:
        return datetime(2026, 1, day, hour, 0, 0, tzinfo=timezone.utc)

    return _at


class KeywordExtractor:
    """Deterministic test extractor: 'message' events about a subject emit one
    fact (subject, lives_in, object) parsed from 'subject:object'. Pure and
    side-effect free, so rebuilds are reproducible."""

    def extract(self, event: Event) -> Sequence[ExtractedFact]:
        if event.type.value != "message" or ":" not in event.content:
            return ()
        subject, _, obj = event.content.partition(":")
        return (
            ExtractedFact(
                subject=subject.strip(),
                predicate="lives_in",
                object=obj.strip(),
                valid_from=event.ts,
                confidence=0.9,
            ),
        )


@pytest.fixture
def extractor():
    return KeywordExtractor()


# Fixed vocabulary the fake embedder counts over. Shared words between two
# texts produce overlapping vectors and therefore high cosine, so retrieval
# behaviour is real (not hand-rigged) while staying deterministic and offline.
FAKE_VOCAB = (
    "alice",
    "bob",
    "lives_in",
    "works_at",
    "likes",
    "berlin",
    "lisbon",
    "acme",
    "tea",
    "climbing",
)


class FakeEmbeddingClient:
    """Bag-of-words embedder over a fixed vocabulary. No network.

    A trailing constant dimension keeps every vector non-zero, so L2
    normalization never divides by zero even for out-of-vocabulary text.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        tokens = text.lower().split()
        counts = [float(tokens.count(word)) for word in FAKE_VOCAB]
        return counts + [1.0]


@pytest.fixture
def fake_embedder():
    return FakeEmbeddingClient()


import json as _json
import re as _re

# Matches a rendered "(subject) -[predicate]-> (object)" edge from the detection
# prompt; the [valid_from=...] suffix uses square brackets and is ignored.
_EDGE_RE = _re.compile(r"\(([^)]*)\) -\[([^\]]*)\]-> \(([^)]*)\)")


class RoutedLLM:
    """Offline LLM stub that answers both pipeline prompts deterministically.

    Routes on the system prompt: the extraction prompt opens with "You extract",
    everything else is treated as detection. Extraction turns recognised
    substrings of the message into facts; detection declares ``supersedes`` when
    the candidate shares a subject+predicate with an existing fact but a
    different object, ``duplicate`` when identical, else ``new``. That is the
    real supersession behaviour without a network call.
    """

    def __init__(self, extract_map: dict[str, list[tuple[str, str, str]]]) -> None:
        self._extract_map = extract_map
        self.systems: list[str] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.systems.append(system)
        if system.startswith("You extract"):
            return self._extract(user)
        return self._detect(user)

    def _extract(self, user: str) -> str:
        low = user.lower()
        facts = [
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "valid_from": "2026-01-01",
                "confidence": 0.9,
            }
            for key, triples in self._extract_map.items()
            if key in low
            for subject, predicate, obj in triples
        ]
        return _json.dumps({"facts": facts})

    def _detect(self, user: str) -> str:
        edges = _EDGE_RE.findall(user)
        if not edges:
            return _json.dumps({"relation": "new", "target": None, "reason": "none"})
        cand, existing = edges[0], edges[1:]
        for index, (subject, predicate, obj) in enumerate(existing):
            if subject == cand[0] and predicate == cand[1]:
                if obj == cand[2]:
                    return _json.dumps(
                        {"relation": "duplicate", "target": index, "reason": "same"}
                    )
                return _json.dumps(
                    {"relation": "supersedes", "target": index, "reason": "changed"}
                )
        return _json.dumps({"relation": "new", "target": None, "reason": "new"})


@pytest.fixture
def routed_llm():
    """Factory: build a RoutedLLM from an {message_substring: [(s, p, o)]} map."""

    def _make(extract_map: dict[str, list[tuple[str, str, str]]]) -> RoutedLLM:
        return RoutedLLM(extract_map)

    return _make
