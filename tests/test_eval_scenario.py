"""The gold dataset must be internally consistent and materialize faithfully."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mneme.domain.events import Actor, EventType
from mneme.eval.dataset import (
    DEFAULT_START,
    GOLD_SCENARIOS,
    materialize,
    total_event_count,
)
from mneme.eval.scenario import (
    Assertion,
    QueryKind,
    Scenario,
    ScenarioEvent,
    ScenarioQuery,
)
from mneme.eval.validate import ScenarioError, validate_all, validate_scenario
from mneme.facts.detector import Relation


def _new(subject: str, predicate: str, obj: str) -> Assertion:
    return Assertion(subject, predicate, obj, Relation.NEW)


# --- the shipped gold ---------------------------------------------------------


def test_every_gold_scenario_validates():
    # Arrange / Act / Assert: validate_all raises if any scenario is inconsistent.
    validate_all(GOLD_SCENARIOS)


def test_gold_has_distinct_ids_and_events():
    ids = [scenario.scenario_id for scenario in GOLD_SCENARIOS]
    assert len(ids) == len(set(ids))
    assert total_event_count() == sum(len(s.events) for s in GOLD_SCENARIOS)


def test_validate_all_rejects_duplicate_ids():
    one = GOLD_SCENARIOS[0]
    with pytest.raises(ScenarioError, match="duplicate scenario_id"):
        validate_all((one, one))


# --- materialization ----------------------------------------------------------


def test_materialize_assigns_consecutive_ids_and_offset_timestamps():
    scenario = GOLD_SCENARIOS[0]

    events = materialize(scenario, first_event_id=100)

    assert [e.event_id for e in events] == list(range(100, 100 + len(scenario.events)))
    for event, scenario_event in zip(events, scenario.events):
        assert event.ts == DEFAULT_START + timedelta(days=scenario_event.day)
        assert event.actor is Actor.USER
        assert event.type is EventType.MESSAGE
        assert event.content == scenario_event.content


def test_materialize_honours_custom_start():
    start = datetime(2030, 6, 1, tzinfo=timezone.utc)
    scenario = GOLD_SCENARIOS[0]

    events = materialize(scenario, start=start)

    assert events[0].ts == start + timedelta(days=scenario.events[0].day)


# --- validator: relation invariants ------------------------------------------


def _scenario(*events: ScenarioEvent, queries: tuple[ScenarioQuery, ...] = ()) -> Scenario:
    return Scenario("probe", tuple(events), queries)


def test_new_on_occupied_slot_is_rejected():
    scenario = _scenario(
        ScenarioEvent(0, "a", _new("alice", "lives_in", "berlin")),
        ScenarioEvent(1, "b", _new("alice", "lives_in", "lisbon")),
    )
    with pytest.raises(ScenarioError, match="NEW but slot already believes"):
        validate_scenario(scenario)


def test_supersedes_without_prior_belief_is_rejected():
    scenario = _scenario(
        ScenarioEvent(
            0, "a", Assertion("alice", "lives_in", "lisbon", Relation.SUPERSEDES)
        ),
    )
    with pytest.raises(ScenarioError, match="no prior belief"):
        validate_scenario(scenario)


def test_duplicate_with_different_object_is_rejected():
    scenario = _scenario(
        ScenarioEvent(0, "a", _new("alice", "lives_in", "berlin")),
        ScenarioEvent(
            1, "b", Assertion("alice", "lives_in", "lisbon", Relation.DUPLICATE)
        ),
    )
    with pytest.raises(ScenarioError, match="DUPLICATE"):
        validate_scenario(scenario)


def test_supersedes_that_does_not_change_belief_is_rejected():
    scenario = _scenario(
        ScenarioEvent(0, "a", _new("alice", "lives_in", "berlin")),
        ScenarioEvent(
            1, "b", Assertion("alice", "lives_in", "berlin", Relation.SUPERSEDES)
        ),
    )
    with pytest.raises(ScenarioError, match="does not change belief"):
        validate_scenario(scenario)


def test_out_of_order_days_are_rejected():
    scenario = _scenario(
        ScenarioEvent(5, "a", _new("alice", "lives_in", "berlin")),
        ScenarioEvent(2, "b"),
    )
    with pytest.raises(ScenarioError, match="precedes earlier event"):
        validate_scenario(scenario)


def test_chatter_does_not_change_belief():
    scenario = _scenario(
        ScenarioEvent(0, "a", _new("alice", "lives_in", "berlin")),
        ScenarioEvent(1, "just chatter"),
        ScenarioEvent(
            2, "b", Assertion("alice", "lives_in", "berlin", Relation.DUPLICATE)
        ),
    )
    validate_scenario(scenario)  # does not raise


# --- validator: query answers -------------------------------------------------


def _two_step() -> tuple[ScenarioEvent, ...]:
    return (
        ScenarioEvent(0, "a", _new("alice", "lives_in", "berlin")),
        ScenarioEvent(
            10, "b", Assertion("alice", "lives_in", "lisbon", Relation.SUPERSEDES)
        ),
    )


def test_wrong_current_answer_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (
            ScenarioQuery(
                QueryKind.CURRENT, "alice", "lives_in", "now?", ("berlin",)
            ),
        ),
    )
    with pytest.raises(ScenarioError, match="gold answer"):
        validate_scenario(scenario)


def test_wrong_evolution_chain_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (
            ScenarioQuery(
                QueryKind.EVOLUTION, "alice", "lives_in", "how?", ("lisbon", "berlin")
            ),
        ),
    )
    with pytest.raises(ScenarioError, match="gold answer"):
        validate_scenario(scenario)


def test_historical_answer_picks_belief_in_force():
    scenario = Scenario(
        "probe",
        _two_step(),
        (
            ScenarioQuery(
                QueryKind.HISTORICAL, "alice", "lives_in", "day 5?", ("berlin",), as_of_day=5
            ),
            ScenarioQuery(
                QueryKind.HISTORICAL, "alice", "lives_in", "day 10?", ("lisbon",), as_of_day=10
            ),
        ),
    )
    validate_scenario(scenario)  # does not raise


def test_wrong_historical_answer_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (
            ScenarioQuery(
                QueryKind.HISTORICAL, "alice", "lives_in", "day 5?", ("lisbon",), as_of_day=5
            ),
        ),
    )
    with pytest.raises(ScenarioError, match="as of day 5"):
        validate_scenario(scenario)


def test_historical_before_first_belief_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (
            ScenarioQuery(
                QueryKind.HISTORICAL, "alice", "lives_in", "day -1?", ("berlin",), as_of_day=-1
            ),
        ),
    )
    with pytest.raises(ScenarioError, match="precedes the first belief"):
        validate_scenario(scenario)


def test_historical_without_as_of_day_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (ScenarioQuery(QueryKind.HISTORICAL, "alice", "lives_in", "when?", ("berlin",)),),
    )
    with pytest.raises(ScenarioError, match="no as_of_day"):
        validate_scenario(scenario)


def test_current_with_as_of_day_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (
            ScenarioQuery(
                QueryKind.CURRENT, "alice", "lives_in", "now?", ("lisbon",), as_of_day=3
            ),
        ),
    )
    with pytest.raises(ScenarioError, match="sets as_of_day"):
        validate_scenario(scenario)


def test_query_targeting_unknown_slot_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (ScenarioQuery(QueryKind.CURRENT, "bob", "works_at", "where?", ("acme",)),),
    )
    with pytest.raises(ScenarioError, match="no event ever asserts"):
        validate_scenario(scenario)


def test_empty_answer_is_rejected():
    scenario = Scenario(
        "probe",
        _two_step(),
        (ScenarioQuery(QueryKind.CURRENT, "alice", "lives_in", "now?", ()),),
    )
    with pytest.raises(ScenarioError, match="empty gold answer"):
        validate_scenario(scenario)
