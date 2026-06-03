"""B3 — a Graphiti-like bitemporal store: the faithful substrate baseline.

Where B1/B2 are straw men (free-text RAG / summary), B3 is the *strong*
baseline: a real bitemporal fact store that preserves history, so it can answer
``historical`` and ``evolution`` as well as MNEME does. Its job is not to lose —
it is to test whether MNEME's substrate is worth it at all. A *tie* against B3 is
the verdict that keeps the substrate question open: at MVP scale a direct
bitemporal graph and an event-sourced projection answer identically, so
event-sourcing's payoff (auditability, rebuild, scale-consistency) is invisible
and unproven here.

What makes B3 a different *substrate*, not just a second copy of MNEME's
Supersede policy:

  * **No event log.** MNEME's facts are a derived, rebuildable projection of an
    append-only log (``FactStore.rebuild``). B3 has no log behind it — the edges
    *are* the store, held in memory and invalidated in place. There is nothing to
    replay and nothing to rebuild from.
  * **Edge invalidation, not chain links.** On a contradiction B3 closes the
    current edge's valid-time and stamps a transaction-time ``invalidated_at``
    (Graphiti's "expired" edge), then adds the new edge. History is read back by
    *valid-time intervals*, not by walking a ``superseded_by`` chain.

Detection is held fixed and perfect — B3 is driven by the same gold relations as
the supersede oracle (NEW / DUPLICATE / REFINES / SUPERSEDES), so the *only*
variable between B3 and supersede is the substrate. That puts B3 on the B0 gate's
structured, exact-match scoring path (``SystemReport``), offline and
deterministic, beside supersede and overwrite — not the LLM-judge path the
free-text baselines need.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from mneme.domain.facts import ExtractedFact
from mneme.eval.dataset import (
    DEFAULT_START,
    GOLD_SCENARIOS,
    candidate_for,
    instant_for,
)
from mneme.eval.harness import QueryOutcome, SystemReport
from mneme.eval.scenario import QueryKind, Scenario, ScenarioQuery
from mneme.eval.validate import validate_all
from mneme.facts.detector import Relation

__all__ = [
    "B3_BITEMPORAL",
    "BitemporalEdge",
    "BitemporalStore",
    "evaluate_bitemporal",
]

# The name B3 is reported under, alongside supersede/overwrite in the B0 gate.
B3_BITEMPORAL = "b3-bitemporal"

# Relations that contradict the current edge on a slot and invalidate it. MNEME's
# Supersede policy closes out the prior fact for both, so B3 matches it: the only
# difference between the two systems is the substrate, never the detection.
_INVALIDATING = (Relation.SUPERSEDES, Relation.REFINES)


@dataclass(frozen=True, slots=True)
class BitemporalEdge:
    """One bitemporal fact edge: a (subject, predicate, object) with two clocks.

    Valid-time is ``[valid_from, valid_to)`` (``valid_to is None`` while in
    force); transaction-time is ``created_at`` to ``invalidated_at`` (``None``
    while live). An edge is *live* until a contradiction invalidates it; an
    invalidated edge is kept, which is what lets B3 answer about the past.
    """

    subject: str
    predicate: str
    object: str
    valid_from: datetime
    created_at: datetime
    valid_to: datetime | None = None
    invalidated_at: datetime | None = None

    @property
    def is_live(self) -> bool:
        """True while this edge is the store's current belief for its slot."""
        return self.invalidated_at is None


class BitemporalStore:
    """An in-memory bitemporal edge store with Graphiti-style invalidation.

    Edges are immutable; "invalidation" replaces a live edge with a closed copy
    rather than mutating it, and the store's edge tuple is rebound, never edited
    in place.
    """

    def __init__(self) -> None:
        self._edges: tuple[BitemporalEdge, ...] = ()

    @property
    def edges(self) -> tuple[BitemporalEdge, ...]:
        """Every edge, live and invalidated, in insertion order."""
        return self._edges

    def integrate(
        self, candidate: ExtractedFact, relation: Relation, *, now: datetime
    ) -> None:
        """Fold one candidate fact in under its gold relation.

        DUPLICATE is dropped (the belief is already held). A contradiction
        (SUPERSEDES/REFINES) first invalidates the slot's live edge, ending its
        valid-time where the new edge begins. Everything else just adds an edge.
        """
        if relation is Relation.DUPLICATE:
            return
        if relation in _INVALIDATING:
            self._invalidate_current(
                candidate.subject,
                candidate.predicate,
                valid_to=candidate.valid_from,
                when=now,
            )
        self._edges = (
            *self._edges,
            BitemporalEdge(
                subject=candidate.subject,
                predicate=candidate.predicate,
                object=candidate.object,
                valid_from=candidate.valid_from,
                created_at=now,
            ),
        )

    def current(self, subject: str, predicate: str) -> tuple[str, ...]:
        """The object of the slot's live edge, or empty if the slot is unknown."""
        for edge in self._slot(subject, predicate):
            if edge.is_live:
                return (edge.object,)
        return ()

    def historical(
        self, subject: str, predicate: str, as_of: datetime
    ) -> tuple[str, ...]:
        """The object whose valid-time interval ``[valid_from, valid_to)`` covers ``as_of``."""
        for edge in self._slot(subject, predicate):
            if edge.valid_from <= as_of and (
                edge.valid_to is None or as_of < edge.valid_to
            ):
                return (edge.object,)
        return ()

    def evolution(self, subject: str, predicate: str) -> tuple[str, ...]:
        """The slot's objects in valid-time order — the full history of the belief."""
        return tuple(edge.object for edge in self._slot(subject, predicate))

    def _slot(self, subject: str, predicate: str) -> list[BitemporalEdge]:
        return sorted(
            (
                edge
                for edge in self._edges
                if edge.subject == subject and edge.predicate == predicate
            ),
            key=lambda edge: edge.valid_from,
        )

    def _invalidate_current(
        self, subject: str, predicate: str, *, valid_to: datetime, when: datetime
    ) -> None:
        self._edges = tuple(
            replace(edge, valid_to=valid_to, invalidated_at=when)
            if edge.is_live and edge.subject == subject and edge.predicate == predicate
            else edge
            for edge in self._edges
        )


def evaluate_bitemporal(
    scenarios: tuple[Scenario, ...] = GOLD_SCENARIOS,
    *,
    start: datetime = DEFAULT_START,
) -> SystemReport:
    """Score B3 across the scenarios on the structured exact-match path.

    Validates the gold first, then ingests each timeline into a fresh store under
    the gold relations and checks every query against the gold answer — the same
    ``SystemReport`` rollup the B0 gate produces for supersede/overwrite.
    """
    validate_all(scenarios)
    outcomes = tuple(
        outcome
        for scenario in scenarios
        for outcome in _evaluate_scenario(scenario, start)
    )
    return SystemReport(B3_BITEMPORAL, outcomes)


def _evaluate_scenario(scenario: Scenario, start: datetime) -> list[QueryOutcome]:
    store = BitemporalStore()
    for event in scenario.events:
        candidate = candidate_for(event, start=start)
        if candidate is None or event.assertion is None:
            continue  # chatter carries no fact to integrate
        store.integrate(
            candidate,
            event.assertion.relation,
            now=instant_for(event.day, start=start),
        )
    return [_run_query(store, scenario, query, start) for query in scenario.queries]


def _run_query(
    store: BitemporalStore, scenario: Scenario, query: ScenarioQuery, start: datetime
) -> QueryOutcome:
    return QueryOutcome(
        scenario_id=scenario.scenario_id,
        kind=query.kind,
        subject=query.subject,
        predicate=query.predicate,
        expected=query.answer,
        actual=_answer(store, query, start),
    )


def _answer(
    store: BitemporalStore, query: ScenarioQuery, start: datetime
) -> tuple[str, ...]:
    if query.kind is QueryKind.CURRENT:
        return store.current(query.subject, query.predicate)
    if query.kind is QueryKind.HISTORICAL:
        assert query.as_of_day is not None  # guaranteed by validate_all
        as_of = instant_for(query.as_of_day, start=start)
        return store.historical(query.subject, query.predicate, as_of)
    return store.evolution(query.subject, query.predicate)
