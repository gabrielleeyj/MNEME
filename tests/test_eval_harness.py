"""The B0 gate: supersede must beat overwrite on history-dependent queries.

These tests pin the gate's headline numbers and the oracle that makes the run
deterministic. The discriminator is structural: holding extraction and judgment
fixed (the gold oracle), the only variable is the storage policy, so any gap
between the two systems is the storage policy's doing.
"""

from __future__ import annotations

import pytest

from mneme.domain.facts import ExtractedFact, Fact
from mneme.eval.dataset import DEFAULT_START, GOLD_SCENARIOS, candidate_for
from mneme.eval.harness import (
    OVERWRITE,
    SUPERSEDE,
    SYSTEMS,
    evaluate,
    format_gate,
    run_gate,
)
from mneme.eval.oracle import OracleError, ScenarioOracleDetector
from mneme.eval.scenario import QueryKind
from mneme.facts.detector import Relation


# --- the gate numbers ---------------------------------------------------------


def test_supersede_answers_everything():
    # The thesis: history-preserving supersession + the gold oracle is perfect.
    report = evaluate(SUPERSEDE)

    assert report.accuracy == 1.0
    assert all(score == 1.0 for score in report.accuracy_by_kind().values())


def test_overwrite_loses_exactly_the_history_queries():
    # The B0 ablation keeps one row per slot: current survives, the past is gone.
    by_kind = evaluate(OVERWRITE).accuracy_by_kind()

    assert by_kind[QueryKind.CURRENT] == 1.0
    assert by_kind[QueryKind.HISTORICAL] == 0.0
    # Only the stable-preference control (a chain of one) survives evolution.
    assert by_kind[QueryKind.EVOLUTION] == pytest.approx(1 / 3)


def test_supersede_strictly_beats_overwrite():
    # The single number the project turns on.
    reports = run_gate()

    assert reports["supersede"].accuracy > reports["overwrite"].accuracy


def test_run_gate_covers_every_system():
    reports = run_gate()

    assert set(reports) == {system.name for system in SYSTEMS}


def test_evaluate_is_deterministic():
    # Fresh in-memory store each call, so two runs must agree exactly.
    first = evaluate(SUPERSEDE)
    second = evaluate(SUPERSEDE)

    assert [o.actual for o in first.outcomes] == [o.actual for o in second.outcomes]


# --- outcome bookkeeping ------------------------------------------------------


def test_every_query_in_every_scenario_is_scored():
    expected = sum(len(scenario.queries) for scenario in GOLD_SCENARIOS)

    assert len(evaluate(SUPERSEDE).outcomes) == expected


def test_outcome_records_expected_and_actual():
    relocation = next(s for s in GOLD_SCENARIOS if s.scenario_id == "alice-relocation")
    historical = next(q for q in relocation.queries if q.kind is QueryKind.HISTORICAL)

    overwrite_out = next(
        o
        for o in evaluate(OVERWRITE).outcomes
        if o.scenario_id == "alice-relocation" and o.kind is QueryKind.HISTORICAL
    )

    assert overwrite_out.expected == historical.answer
    assert overwrite_out.actual != historical.answer
    assert not overwrite_out.correct


def test_empty_report_has_zero_accuracy():
    report = evaluate(SUPERSEDE, scenarios=())

    assert report.accuracy == 0.0
    assert report.accuracy_by_kind() == {}


# --- the gold oracle ----------------------------------------------------------


def test_oracle_replays_the_authored_relation():
    relocation = next(s for s in GOLD_SCENARIOS if s.scenario_id == "alice-relocation")
    oracle = ScenarioOracleDetector(relocation)

    first = relocation.events[0]  # alice:berlin, NEW
    candidate = candidate_for(first)
    judgment = oracle.judge(candidate, existing=())

    assert judgment.relation is Relation.NEW
    assert judgment.target_fact_id is None


def test_oracle_rejects_an_unknown_candidate():
    relocation = next(s for s in GOLD_SCENARIOS if s.scenario_id == "alice-relocation")
    oracle = ScenarioOracleDetector(relocation)

    stranger = ExtractedFact(
        subject="nobody",
        predicate="from",
        object="nowhere",
        valid_from=DEFAULT_START,
    )

    with pytest.raises(OracleError):
        oracle.judge(stranger, existing=())


def test_oracle_rejects_a_targeted_relation_with_no_current_fact():
    # A SUPERSEDES candidate needs a current fact on its slot; absent one, the
    # oracle refuses rather than inventing a target.
    relocation = next(s for s in GOLD_SCENARIOS if s.scenario_id == "alice-relocation")
    oracle = ScenarioOracleDetector(relocation)

    lisbon = next(
        e
        for e in relocation.events
        if e.assertion is not None and e.assertion.relation is Relation.SUPERSEDES
    )
    candidate = candidate_for(lisbon)

    with pytest.raises(OracleError):
        oracle.judge(candidate, existing=())


# --- rendering ----------------------------------------------------------------


def test_format_gate_lists_every_system_and_kind():
    rendered = format_gate(run_gate())

    assert "supersede" in rendered
    assert "overwrite" in rendered
    for kind in QueryKind:
        assert kind.value in rendered
