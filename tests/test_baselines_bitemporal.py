"""B3 bitemporal store: edge invalidation, interval queries, and scored rollups.

Fully offline and deterministic — B3 makes no LLM calls at all. It is driven by
the gold relations (the same ones the supersede oracle replays), so the only
variable between B3 and supersede is the substrate, and the scored run is exact
arithmetic, not a judged sample.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mneme.baselines.bitemporal import (
    B3_BITEMPORAL,
    BitemporalStore,
    evaluate_bitemporal,
)
from mneme.domain.facts import ExtractedFact
from mneme.eval.scenario import QueryKind
from mneme.facts.detector import Relation

_START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(n: int) -> datetime:
    return _START + timedelta(days=n)


def _candidate(subject: str, predicate: str, obj: str, *, day: int) -> ExtractedFact:
    return ExtractedFact(subject, predicate, obj, valid_from=_day(day))


# --- the bitemporal store -----------------------------------------------------


def test_new_fact_becomes_the_live_current_belief():
    store = BitemporalStore()

    store.integrate(_candidate("alice", "lives_in", "berlin", day=0), Relation.NEW, now=_day(0))

    assert store.current("alice", "lives_in") == ("berlin",)
    (edge,) = store.edges
    assert edge.is_live
    assert edge.valid_to is None and edge.invalidated_at is None


def test_supersedes_invalidates_the_prior_edge_and_keeps_it():
    store = BitemporalStore()
    store.integrate(_candidate("alice", "lives_in", "berlin", day=0), Relation.NEW, now=_day(0))

    store.integrate(
        _candidate("alice", "lives_in", "lisbon", day=40), Relation.SUPERSEDES, now=_day(40)
    )

    # Current flips to the new belief; the old edge is kept but closed out.
    assert store.current("alice", "lives_in") == ("lisbon",)
    old, new = store.edges
    assert old.object == "berlin"
    assert not old.is_live
    assert old.valid_to == _day(40)  # valid-time ends where the new one begins
    assert old.invalidated_at == _day(40)  # transaction-time close-out
    assert new.is_live and new.object == "lisbon"


def test_evolution_lists_the_slot_in_valid_time_order():
    store = BitemporalStore()
    store.integrate(_candidate("bob", "works_at", "acme", day=0), Relation.NEW, now=_day(0))
    store.integrate(_candidate("bob", "works_at", "acme corp", day=5), Relation.REFINES, now=_day(5))
    store.integrate(_candidate("bob", "works_at", "globex", day=30), Relation.SUPERSEDES, now=_day(30))

    assert store.evolution("bob", "works_at") == ("acme", "acme corp", "globex")
    assert store.current("bob", "works_at") == ("globex",)


def test_refines_keeps_both_edges_like_supersedes():
    # MNEME closes out the prior fact for REFINES too, so B3 must keep both.
    store = BitemporalStore()
    store.integrate(_candidate("bob", "works_at", "acme", day=0), Relation.NEW, now=_day(0))

    store.integrate(_candidate("bob", "works_at", "acme corp", day=5), Relation.REFINES, now=_day(5))

    assert store.evolution("bob", "works_at") == ("acme", "acme corp")
    assert len(store.edges) == 2


def test_historical_returns_the_edge_in_force_at_an_instant():
    store = BitemporalStore()
    store.integrate(_candidate("alice", "lives_in", "berlin", day=0), Relation.NEW, now=_day(0))
    store.integrate(_candidate("alice", "lives_in", "lisbon", day=40), Relation.SUPERSEDES, now=_day(40))

    assert store.historical("alice", "lives_in", _day(20)) == ("berlin",)  # before the move
    assert store.historical("alice", "lives_in", _day(40)) == ("lisbon",)  # boundary is half-open
    assert store.historical("alice", "lives_in", _day(50)) == ("lisbon",)


def test_duplicate_is_a_no_op():
    store = BitemporalStore()
    store.integrate(_candidate("alice", "likes", "tea", day=2), Relation.NEW, now=_day(2))

    store.integrate(_candidate("alice", "likes", "tea", day=15), Relation.DUPLICATE, now=_day(15))

    assert len(store.edges) == 1
    assert store.evolution("alice", "likes") == ("tea",)


def test_unknown_slot_returns_empty():
    store = BitemporalStore()

    assert store.current("nobody", "lives_in") == ()
    assert store.historical("nobody", "lives_in", _day(5)) == ()
    assert store.evolution("nobody", "lives_in") == ()


def test_integrate_does_not_mutate_existing_edges():
    store = BitemporalStore()
    store.integrate(_candidate("alice", "lives_in", "berlin", day=0), Relation.NEW, now=_day(0))
    (before,) = store.edges

    store.integrate(_candidate("alice", "lives_in", "lisbon", day=40), Relation.SUPERSEDES, now=_day(40))

    # The original frozen edge object is untouched; invalidation produced a copy.
    assert before.is_live
    assert before.valid_to is None


# --- the scored harness (structured exact-match path) -------------------------


def test_evaluate_bitemporal_scores_every_gold_query():
    from mneme.eval.dataset import GOLD_SCENARIOS

    expected = sum(len(scenario.queries) for scenario in GOLD_SCENARIOS)
    report = evaluate_bitemporal()

    assert report.system == B3_BITEMPORAL
    assert len(report.outcomes) == expected


def test_evaluate_bitemporal_ties_supersede_at_full_accuracy():
    report = evaluate_bitemporal()

    assert report.accuracy == 1.0
    by_kind = report.accuracy_by_kind()
    assert by_kind[QueryKind.CURRENT] == 1.0
    assert by_kind[QueryKind.HISTORICAL] == 1.0
    assert by_kind[QueryKind.EVOLUTION] == 1.0


def test_evaluate_bitemporal_answers_the_overwrite_discriminators():
    # The exact queries overwrite gets wrong, B3 gets right — because it kept history.
    report = evaluate_bitemporal()
    by_query = {(o.scenario_id, o.kind): o for o in report.outcomes}

    bob_hist = by_query[("bob-career", QueryKind.HISTORICAL)]
    assert bob_hist.actual == ("acme corp",)

    bob_evo = by_query[("bob-career", QueryKind.EVOLUTION)]
    assert bob_evo.actual == ("acme", "acme corp", "globex")

    alice_evo = by_query[("alice-relocation", QueryKind.EVOLUTION)]
    assert alice_evo.actual == ("berlin", "lisbon")


def test_empty_report_has_zero_accuracy():
    report = evaluate_bitemporal(scenarios=())

    assert report.accuracy == 0.0
    assert report.accuracy_by_kind() == {}
