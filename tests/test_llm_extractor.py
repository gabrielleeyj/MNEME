from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mneme.domain.events import Actor, Event, EventType
from mneme.facts.llm_extractor import ExtractionError, LLMExtractor


class FakeLLMClient:
    """Records calls and returns a canned response. No network."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


def _event(content: str, *, day: int = 1) -> Event:
    return Event(
        event_id=1,
        ts=datetime(2026, 1, day, 12, 0, 0, tzinfo=timezone.utc),
        actor=Actor.USER,
        type=EventType.MESSAGE,
        content=content,
    )


def test_extracts_single_fact_with_explicit_valid_from():
    client = FakeLLMClient(
        '{"facts": [{"subject": "alice", "predicate": "lives_in", '
        '"object": "Berlin", "valid_from": "2026-01-05", "confidence": 0.9}]}'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("I moved to Berlin on Jan 5"))

    assert len(facts) == 1
    fact = facts[0]
    assert (fact.subject, fact.predicate, fact.object) == ("alice", "lives_in", "Berlin")
    assert fact.valid_from == datetime(2026, 1, 5, tzinfo=timezone.utc)
    assert fact.confidence == 0.9


def test_valid_from_defaults_to_event_timestamp_when_null():
    client = FakeLLMClient(
        '{"facts": [{"subject": "bob", "predicate": "likes", '
        '"object": "tea", "valid_from": null}]}'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("I like tea", day=7))

    assert facts[0].valid_from == datetime(2026, 1, 7, 12, 0, 0, tzinfo=timezone.utc)


def test_valid_from_defaults_when_key_absent():
    client = FakeLLMClient(
        '{"facts": [{"subject": "bob", "predicate": "likes", "object": "tea"}]}'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("I like tea", day=9))

    assert facts[0].valid_from == datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
    assert facts[0].confidence is None


def test_naive_valid_from_is_coerced_to_utc():
    client = FakeLLMClient(
        '{"facts": [{"subject": "a", "predicate": "p", "object": "o", '
        '"valid_from": "2026-02-01T08:30:00"}]}'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("x"))

    assert facts[0].valid_from == datetime(2026, 2, 1, 8, 30, tzinfo=timezone.utc)


def test_extracts_multiple_facts():
    client = FakeLLMClient(
        '{"facts": ['
        '{"subject": "alice", "predicate": "lives_in", "object": "Berlin"},'
        '{"subject": "alice", "predicate": "works_at", "object": "Acme"}'
        ']}'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("I moved to Berlin and joined Acme"))

    assert {(f.predicate, f.object) for f in facts} == {
        ("lives_in", "Berlin"),
        ("works_at", "Acme"),
    }


def test_no_facts_returns_empty_sequence():
    client = FakeLLMClient('{"facts": []}')
    extractor = LLMExtractor(client)

    assert extractor.extract(_event("hello there")) == ()


def test_strips_markdown_code_fences():
    client = FakeLLMClient(
        '```json\n{"facts": [{"subject": "a", "predicate": "p", "object": "o"}]}\n```'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("x"))

    assert len(facts) == 1


def test_recovers_json_embedded_in_prose():
    client = FakeLLMClient(
        'Here are the facts I found:\n'
        '{"facts": [{"subject": "a", "predicate": "p", "object": "o"}]}\n'
        'Hope that helps!'
    )
    extractor = LLMExtractor(client)

    facts = extractor.extract(_event("x"))

    assert len(facts) == 1


def test_malformed_json_raises_extraction_error():
    client = FakeLLMClient("not json at all")
    extractor = LLMExtractor(client)

    with pytest.raises(ExtractionError):
        extractor.extract(_event("x"))


def test_missing_facts_key_raises_extraction_error():
    client = FakeLLMClient('{"results": []}')
    extractor = LLMExtractor(client)

    with pytest.raises(ExtractionError):
        extractor.extract(_event("x"))


def test_fact_missing_required_field_raises_extraction_error():
    client = FakeLLMClient(
        '{"facts": [{"subject": "a", "predicate": "p"}]}'  # no object
    )
    extractor = LLMExtractor(client)

    with pytest.raises(ExtractionError):
        extractor.extract(_event("x"))


def test_blank_required_field_raises_extraction_error():
    client = FakeLLMClient(
        '{"facts": [{"subject": "  ", "predicate": "p", "object": "o"}]}'
    )
    extractor = LLMExtractor(client)

    with pytest.raises(ExtractionError):
        extractor.extract(_event("x"))


def test_event_content_and_timestamp_reach_the_prompt():
    client = FakeLLMClient('{"facts": []}')
    extractor = LLMExtractor(client)

    extractor.extract(_event("unique-marker-text", day=3))

    assert len(client.calls) == 1
    user_prompt = client.calls[0]["user"]
    assert "unique-marker-text" in user_prompt
    assert "2026-01-03" in user_prompt
    assert client.calls[0]["system"]  # a non-empty system prompt was sent


def test_satisfies_extractor_protocol():
    from mneme.facts.extractor import Extractor

    extractor = LLMExtractor(FakeLLMClient('{"facts": []}'))
    assert isinstance(extractor, Extractor)
