from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mneme.domain.facts import ExtractedFact, Fact
from mneme.facts.detector import (
    ContradictionDetector,
    DetectionError,
    Detector,
    Relation,
    SlotDetector,
)


class FakeLLMClient:
    """Records calls and returns a canned response. No network."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


class ExplodingLLMClient:
    """Fails if called — proves a code path never reaches the model."""

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        raise AssertionError("the LLM must not be called")


def _candidate(obj: str = "Lisbon") -> ExtractedFact:
    return ExtractedFact(
        subject="alice",
        predicate="lives_in",
        object=obj,
        valid_from=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )


def _fact(fact_id: int, obj: str = "Berlin", *, subject: str = "alice") -> Fact:
    return Fact(
        fact_id=fact_id,
        subject=subject,
        predicate="lives_in",
        object=obj,
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_event_id=1,
    )


def test_no_existing_short_circuits_to_new_without_calling_llm():
    detector = ContradictionDetector(ExplodingLLMClient())
    judgment = detector.judge(_candidate(), [])
    assert judgment.relation is Relation.NEW
    assert judgment.target_fact_id is None


def test_supersedes_maps_target_index_to_fact_id():
    client = FakeLLMClient('{"relation": "supersedes", "target": 1, "reason": "moved"}')
    detector = ContradictionDetector(client)

    judgment = detector.judge(_candidate("Lisbon"), [_fact(10, "Munich"), _fact(20, "Berlin")])

    assert judgment.relation is Relation.SUPERSEDES
    assert judgment.target_fact_id == 20
    assert judgment.reason == "moved"


def test_refines_is_parsed():
    client = FakeLLMClient('{"relation": "refines", "target": 0, "reason": "sharper"}')
    detector = ContradictionDetector(client)
    judgment = detector.judge(_candidate("Berlin"), [_fact(7, "Germany")])
    assert judgment.relation is Relation.REFINES
    assert judgment.target_fact_id == 7


def test_duplicate_is_parsed():
    client = FakeLLMClient('{"relation": "duplicate", "target": 0, "reason": "same"}')
    detector = ContradictionDetector(client)
    judgment = detector.judge(_candidate("Berlin"), [_fact(7, "Berlin")])
    assert judgment.relation is Relation.DUPLICATE
    assert judgment.target_fact_id == 7


def test_new_with_null_target():
    client = FakeLLMClient('{"relation": "new", "target": null, "reason": "unrelated"}')
    detector = ContradictionDetector(client)
    judgment = detector.judge(_candidate(), [_fact(7, "Berlin", subject="bob")])
    assert judgment.relation is Relation.NEW
    assert judgment.target_fact_id is None


def test_strips_code_fences_around_response():
    client = FakeLLMClient(
        '```json\n{"relation": "supersedes", "target": 0, "reason": "x"}\n```'
    )
    detector = ContradictionDetector(client)
    judgment = detector.judge(_candidate(), [_fact(42)])
    assert judgment.target_fact_id == 42


def test_unknown_relation_raises():
    client = FakeLLMClient('{"relation": "contradicts", "target": 0}')
    detector = ContradictionDetector(client)
    with pytest.raises(DetectionError):
        detector.judge(_candidate(), [_fact(1)])


def test_targeted_relation_without_target_raises():
    client = FakeLLMClient('{"relation": "supersedes", "target": null}')
    detector = ContradictionDetector(client)
    with pytest.raises(DetectionError):
        detector.judge(_candidate(), [_fact(1)])


def test_out_of_range_target_raises():
    client = FakeLLMClient('{"relation": "supersedes", "target": 5}')
    detector = ContradictionDetector(client)
    with pytest.raises(DetectionError):
        detector.judge(_candidate(), [_fact(1)])


def test_boolean_target_raises():
    # JSON true is an int subclass in Python; it must not be accepted as index 1.
    client = FakeLLMClient('{"relation": "supersedes", "target": true}')
    detector = ContradictionDetector(client)
    with pytest.raises(DetectionError):
        detector.judge(_candidate(), [_fact(1), _fact(2)])


def test_malformed_json_raises():
    client = FakeLLMClient("not json")
    detector = ContradictionDetector(client)
    with pytest.raises(DetectionError):
        detector.judge(_candidate(), [_fact(1)])


def test_candidate_and_existing_reach_the_prompt():
    client = FakeLLMClient('{"relation": "supersedes", "target": 0, "reason": "x"}')
    detector = ContradictionDetector(client)

    detector.judge(_candidate("Lisbon"), [_fact(1, "Berlin")])

    user_prompt = client.calls[0]["user"]
    assert "Lisbon" in user_prompt  # the candidate
    assert "Berlin" in user_prompt  # the existing fact
    assert client.calls[0]["system"]  # a non-empty system prompt was sent


def test_contradiction_detector_satisfies_protocol():
    detector = ContradictionDetector(FakeLLMClient('{"relation": "new"}'))
    assert isinstance(detector, Detector)


# --- SlotDetector (deterministic, keyless) -------------------------------------


def _other(fact_id: int, predicate: str, obj: str) -> Fact:
    return Fact(
        fact_id=fact_id,
        subject="alice",
        predicate=predicate,
        object=obj,
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_event_id=1,
    )


def test_slot_detector_new_when_no_facts():
    judgment = SlotDetector().judge(_candidate(), [])
    assert judgment.relation is Relation.NEW
    assert judgment.target_fact_id is None


def test_slot_detector_new_when_no_same_predicate():
    # A fact about the same subject but a different slot is not a conflict.
    judgment = SlotDetector().judge(_candidate("Lisbon"), [_other(7, "works_at", "Acme")])
    assert judgment.relation is Relation.NEW


def test_slot_detector_duplicate_on_identical_value():
    judgment = SlotDetector().judge(_candidate("Berlin"), [_fact(3, "Berlin")])
    assert judgment.relation is Relation.DUPLICATE
    assert judgment.target_fact_id == 3


def test_slot_detector_supersedes_on_changed_value():
    judgment = SlotDetector().judge(_candidate("Lisbon"), [_fact(5, "Berlin")])
    assert judgment.relation is Relation.SUPERSEDES
    assert judgment.target_fact_id == 5


def test_slot_detector_satisfies_protocol():
    assert isinstance(SlotDetector(), Detector)
