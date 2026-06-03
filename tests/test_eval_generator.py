"""The procedural scale generator: determinism, shape, and the offline gate.

Fully offline and deterministic — the generator makes no LLM calls, and every
scenario is derived from a seeded RNG, so a given ``GenConfig`` always yields the
same events and queries. The point of the dataset is to be the *discriminator*:
a long timeline where overwrite collapses on history while supersede (and B3)
hold, so the gate test asserts exactly that separation.
"""

from __future__ import annotations

from collections import Counter

from mneme.baselines.bitemporal import B3_BITEMPORAL, evaluate_bitemporal
from mneme.eval.generator import (
    GenConfig,
    default_config,
    generate,
    scale_config,
)
from mneme.eval.harness import run_gate
from mneme.eval.scenario import QueryKind
from mneme.eval.validate import validate_all, validate_scenario


# --- determinism --------------------------------------------------------------


def test_same_config_yields_identical_scenarios():
    one = generate(scale_config())
    two = generate(scale_config())

    assert [(e.day, e.content) for e in one.events] == [
        (e.day, e.content) for e in two.events
    ]
    assert [q.answer for q in one.queries] == [q.answer for q in two.queries]


def test_different_seed_yields_a_different_scenario():
    one = generate(GenConfig(seed=1, num_subjects=6, scenario_id="a"))
    two = generate(GenConfig(seed=2, num_subjects=6, scenario_id="a"))

    assert [e.content for e in one.events] != [e.content for e in two.events]


# --- shape and validity -------------------------------------------------------


def test_generated_scenario_passes_its_own_validation():
    # generate() runs validate_scenario internally; this is the explicit assert.
    validate_scenario(generate(default_config()))
    validate_scenario(generate(scale_config()))


def test_generated_scenario_is_accepted_by_validate_all():
    scenario = generate(scale_config())
    validate_all((scenario,))  # unique id, internally consistent


def test_default_config_is_small_and_fast():
    scenario = generate(default_config())

    assert scenario.scenario_id == "scale-small"
    assert len(scenario.events) < 200


def test_scale_config_is_a_long_timeline():
    scenario = generate(scale_config())

    # The headline is "~1000 events"; assert a generous band around it.
    assert 900 <= len(scenario.events) <= 1300


def test_scale_scenario_exercises_all_three_query_kinds():
    scenario = generate(scale_config())
    kinds = Counter(query.kind for query in scenario.queries)

    assert kinds[QueryKind.CURRENT] > 0
    assert kinds[QueryKind.HISTORICAL] > 0
    assert kinds[QueryKind.EVOLUTION] > 0


def test_scale_scenario_has_multi_step_histories():
    scenario = generate(scale_config())

    evolutions = [q for q in scenario.queries if q.kind is QueryKind.EVOLUTION]
    assert evolutions, "expected evolution queries on rich slots"
    # At least one slot changed belief more than once — the thing overwrite loses.
    assert any(len(q.answer) >= 2 for q in evolutions)


def test_scale_scenario_is_buried_in_chatter():
    scenario = generate(scale_config())

    chatter = [e for e in scenario.events if e.assertion is None]
    facts = [e for e in scenario.events if e.assertion is not None]
    assert chatter, "expected non-fact chatter events"
    # Chatter should be a meaningful fraction — the retrieval/summary pressure.
    assert len(chatter) >= 0.3 * len(facts)


def test_events_are_globally_non_decreasing_in_day():
    events = generate(scale_config()).events

    days = [event.day for event in events]
    assert days == sorted(days)


# --- the offline scale gate (the discriminator) -------------------------------


def test_overwrite_collapses_on_history_at_scale():
    scenario = generate(scale_config())
    reports = run_gate(scenarios=(scenario,))

    supersede = reports["supersede"]
    overwrite = reports["overwrite"]

    # Supersede answers everything; overwrite keeps only the present value.
    assert supersede.accuracy == 1.0
    by_kind = overwrite.accuracy_by_kind()
    assert by_kind[QueryKind.CURRENT] == 1.0
    assert by_kind[QueryKind.HISTORICAL] == 0.0
    assert by_kind[QueryKind.EVOLUTION] == 0.0


def test_b3_ties_supersede_at_scale():
    scenario = generate(scale_config())
    report = evaluate_bitemporal(scenarios=(scenario,))

    assert report.system == B3_BITEMPORAL
    assert report.accuracy == 1.0
    by_kind = report.accuracy_by_kind()
    assert by_kind[QueryKind.CURRENT] == 1.0
    assert by_kind[QueryKind.HISTORICAL] == 1.0
    assert by_kind[QueryKind.EVOLUTION] == 1.0
