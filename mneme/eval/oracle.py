"""An oracle detector — the gold relations, standing in for the LLM judge.

The B0 ablation asks whether *the storage policy* is worth it: given correct
extraction and correct judgment, does history-preserving supersession beat
last-write-wins overwrite? To measure that cleanly the harness must hold
judgment fixed and perfect, so it swaps the LLM ``ContradictionDetector`` for
this oracle, which replays the relation each scenario already declares.

Detector quality (the real model's false-supersession rate) is a separate
question that needs the LLM and the same gold as its answer key; it is not what
the oracle measures.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from mneme.domain.facts import ExtractedFact, Fact
from mneme.eval.dataset import DEFAULT_START, candidate_for
from mneme.eval.scenario import Scenario
from mneme.facts.detector import Judgment, Relation

__all__ = ["OracleError", "ScenarioOracleDetector"]


class OracleError(ValueError):
    """The oracle was asked to judge a candidate the scenario never asserts."""


class ScenarioOracleDetector:
    """A ``Detector`` that returns a scenario's authored relation per candidate.

    Keyed by the exact candidate, so a duplicate restatement (same triple, later
    ``valid_from``) resolves to its own gold relation rather than colliding with
    the original assertion.
    """

    def __init__(self, scenario: Scenario, *, start: datetime = DEFAULT_START) -> None:
        self._relations: dict[ExtractedFact, Relation] = {}
        for event in scenario.events:
            candidate = candidate_for(event, start=start)
            if candidate is not None:
                self._relations[candidate] = event.assertion.relation

    def judge(self, candidate: ExtractedFact, existing: Sequence[Fact]) -> Judgment:
        relation = self._relations.get(candidate)
        if relation is None:
            raise OracleError(f"no gold relation for candidate {candidate!r}")

        if relation is Relation.NEW:
            return Judgment(Relation.NEW, None, "gold: new")

        target = _current_target(candidate, existing)
        if target is None:
            raise OracleError(
                f"gold relation {relation.value} for {candidate.subject!r}/"
                f"{candidate.predicate!r} has no current fact to act on"
            )
        return Judgment(relation, target, f"gold: {relation.value}")


def _current_target(candidate: ExtractedFact, existing: Sequence[Fact]) -> int | None:
    """The current fact on the candidate's slot, the one a non-NEW relation acts on."""
    for fact in existing:
        if (
            fact.is_current
            and fact.subject == candidate.subject
            and fact.predicate == candidate.predicate
        ):
            return fact.fact_id
    return None
