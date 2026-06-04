"""The contradiction detector — the thesis and the risk (workstream 3).

Given a new candidate fact and the existing facts it might touch, classify the
relationship: is it genuinely NEW, a DUPLICATE of something already believed, a
REFINES that sharpens an existing belief, or a SUPERSEDES that contradicts and
replaces one because the truth changed?

The hard, project-defining error is a *false supersession*: closing out a fact
that was not really contradicted. So the judge is tuned for precision and
short-circuits to NEW when there is nothing to compare against — no LLM call,
no chance to hallucinate a conflict.

The response is untrusted external data, validated strictly: an unknown
relation, or a non-NEW verdict that points at no existing fact (or an
out-of-range one), raises ``DetectionError`` rather than guessing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from mneme.domain.facts import ExtractedFact, Fact
from mneme.facts.detection_prompts import (
    DETECTION_SYSTEM_PROMPT,
    build_detection_user_prompt,
)
from mneme.llm.client import LLMClient
from mneme.llm.json_io import JSONExtractionError, extract_json_object


class Relation(str, Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    REFINES = "refines"
    SUPERSEDES = "supersedes"


# Relations that must name an existing fact they act on.
_TARGETED = (Relation.DUPLICATE, Relation.REFINES, Relation.SUPERSEDES)


@dataclass(frozen=True, slots=True)
class Judgment:
    """How a candidate fact relates to the existing facts.

    ``target_fact_id`` is the existing fact the candidate duplicates, refines, or
    supersedes; ``None`` exactly when the relation is ``NEW``.
    """

    relation: Relation
    target_fact_id: int | None
    reason: str


class DetectionError(ValueError):
    """The LLM response could not be parsed into a well-formed judgment."""


@runtime_checkable
class Detector(Protocol):
    def judge(
        self, candidate: ExtractedFact, existing: Sequence[Fact]
    ) -> Judgment:
        """Classify how ``candidate`` relates to the ``existing`` facts."""
        ...


class SlotDetector:
    """A deterministic, LLM-free judge for already-extracted triples.

    The keyless fallback: when no API key is set, the *host* agent does the
    prose-to-triple extraction and hands MNEME clean ``(subject, predicate,
    object)`` facts. With the hard part already done, conflict detection reduces
    to an exact slot rule, no model required —

      * an existing current fact for the same subject+predicate with the same
        object is a ``DUPLICATE`` (the belief is already held);
      * one with a *different* object is a ``SUPERSEDES`` (the slot's value
        changed), closed out so history is preserved exactly as the LLM path
        would;
      * anything else is ``NEW``.

    Supersession keeps at most one current fact per slot, so the first
    same-slot match is the one to close out. Implements the ``Detector``
    protocol, so it drops into ``SupersedePolicy`` unchanged.
    """

    def judge(self, candidate: ExtractedFact, existing: Sequence[Fact]) -> Judgment:
        same_slot = [f for f in existing if f.predicate == candidate.predicate]
        for fact in same_slot:
            if fact.object == candidate.object:
                return Judgment(
                    Relation.DUPLICATE, fact.fact_id, "same value already current"
                )
        if same_slot:
            return Judgment(
                Relation.SUPERSEDES,
                same_slot[0].fact_id,
                "new value for an existing subject+predicate slot",
            )
        return Judgment(Relation.NEW, None, "no current fact for this slot")


class ContradictionDetector:
    def __init__(self, client: LLMClient, *, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def judge(
        self, candidate: ExtractedFact, existing: Sequence[Fact]
    ) -> Judgment:
        if not existing:
            return Judgment(Relation.NEW, None, "no existing facts to compare against")

        raw = self._client.complete(
            system=DETECTION_SYSTEM_PROMPT,
            user=build_detection_user_prompt(candidate, existing),
            max_tokens=self._max_tokens,
        )
        return _parse_judgment(raw, existing)


def _parse_judgment(raw: str, existing: Sequence[Fact]) -> Judgment:
    try:
        data = extract_json_object(raw)
    except JSONExtractionError as exc:
        raise DetectionError(str(exc)) from exc
    if not isinstance(data, dict):
        raise DetectionError(f"response is not a JSON object: {raw!r}")

    relation = _parse_relation(data.get("relation"), raw)
    reason = data.get("reason")
    reason_text = reason.strip() if isinstance(reason, str) else ""

    if relation is Relation.NEW:
        return Judgment(Relation.NEW, None, reason_text)

    target_fact_id = _resolve_target(data.get("target"), existing, raw)
    return Judgment(relation, target_fact_id, reason_text)


def _parse_relation(raw_relation: Any, raw: str) -> Relation:
    if not isinstance(raw_relation, str):
        raise DetectionError(f"response missing string 'relation': {raw!r}")
    try:
        return Relation(raw_relation.strip().lower())
    except ValueError as exc:
        raise DetectionError(f"unknown relation {raw_relation!r}: {raw!r}") from exc


def _resolve_target(raw_target: Any, existing: Sequence[Fact], raw: str) -> int:
    if not isinstance(raw_target, int) or isinstance(raw_target, bool):
        raise DetectionError(
            f"relation requires an integer 'target' index, got {raw_target!r}: {raw!r}"
        )
    if not 0 <= raw_target < len(existing):
        raise DetectionError(
            f"'target' index {raw_target} out of range for {len(existing)} facts: {raw!r}"
        )
    return existing[raw_target].fact_id
