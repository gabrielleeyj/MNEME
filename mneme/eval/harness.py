"""The eval harness — the B0 gate, run offline against the gold dataset.

For each scenario and each system, ingest the timeline into a fresh in-memory
store, then ask the query router every gold question and check the answer. Two
systems are compared:

  * ``supersede`` — the thesis (``SupersedePolicy`` + the oracle detector,
    keeping full history).
  * ``overwrite`` — the B0 ablation (``OverwritePolicy``, one row per slot).

Both run the *same* gold extraction and the *same* gold judgment, so the only
variable is the storage policy. The number that matters is the gap between them
on ``historical`` and ``evolution`` queries — what overwrite cannot answer once
it has thrown the past away.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from mneme.db import init_db
from mneme.domain.events import Actor, EventType
from mneme.eval.dataset import DEFAULT_START, GOLD_SCENARIOS, candidate_for, instant_for
from mneme.eval.oracle import ScenarioOracleDetector
from mneme.eval.scenario import QueryKind, Scenario, ScenarioQuery
from mneme.eval.validate import validate_all
from mneme.facts.candidates import SubjectCandidateProvider
from mneme.facts.policy import OverwritePolicy, SupersedePolicy, WritePolicy
from mneme.facts.store import FactStore
from mneme.log.event_log import EventLog
from mneme.query.router import QueryRouter

__all__ = [
    "QueryOutcome",
    "SystemReport",
    "System",
    "SUPERSEDE",
    "OVERWRITE",
    "SYSTEMS",
    "evaluate",
    "run_gate",
    "format_gate",
]


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One gold query under one system: what was expected, what came back."""

    scenario_id: str
    kind: QueryKind
    subject: str
    predicate: str
    expected: tuple[str, ...]
    actual: tuple[str, ...]

    @property
    def correct(self) -> bool:
        return self.actual == self.expected


@dataclass(frozen=True, slots=True)
class SystemReport:
    """Every outcome for one system, with accuracy rollups."""

    system: str
    outcomes: tuple[QueryOutcome, ...]

    @property
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.correct for o in self.outcomes) / len(self.outcomes)

    def accuracy_by_kind(self) -> dict[QueryKind, float]:
        scores: dict[QueryKind, float] = {}
        for kind in QueryKind:
            of_kind = [o for o in self.outcomes if o.kind is kind]
            if of_kind:
                scores[kind] = sum(o.correct for o in of_kind) / len(of_kind)
        return scores


@dataclass(frozen=True, slots=True)
class System:
    """A named storage policy under test, built fresh per scenario."""

    name: str
    build_policy: Callable[[sqlite3.Connection, Scenario], WritePolicy]


SUPERSEDE = System(
    "supersede",
    lambda conn, scenario: SupersedePolicy(
        ScenarioOracleDetector(scenario), SubjectCandidateProvider(conn)
    ),
)
OVERWRITE = System("overwrite", lambda conn, scenario: OverwritePolicy())
SYSTEMS: tuple[System, ...] = (SUPERSEDE, OVERWRITE)


def evaluate(
    system: System,
    scenarios: tuple[Scenario, ...] = GOLD_SCENARIOS,
    *,
    start: datetime = DEFAULT_START,
) -> SystemReport:
    """Score one system across the scenarios. Validates the gold first."""
    validate_all(scenarios)
    outcomes = tuple(
        outcome
        for scenario in scenarios
        for outcome in _evaluate_scenario(system, scenario, start)
    )
    return SystemReport(system.name, outcomes)


def run_gate(
    scenarios: tuple[Scenario, ...] = GOLD_SCENARIOS,
    *,
    start: datetime = DEFAULT_START,
) -> dict[str, SystemReport]:
    """Score every system; the supersede-vs-overwrite gap is the B0 result."""
    return {system.name: evaluate(system, scenarios, start=start) for system in SYSTEMS}


def _evaluate_scenario(
    system: System, scenario: Scenario, start: datetime
) -> list[QueryOutcome]:
    conn = init_db(":memory:")
    try:
        store = _ingest(system, scenario, conn, start)
        router = QueryRouter(store)
        return [_run_query(router, scenario, query, start) for query in scenario.queries]
    finally:
        conn.close()


def _ingest(
    system: System, scenario: Scenario, conn: sqlite3.Connection, start: datetime
) -> FactStore:
    log = EventLog(conn)
    store = FactStore(conn)
    policy = system.build_policy(conn, scenario)
    for event in scenario.events:
        logged = log.append(
            Actor.USER,
            EventType.MESSAGE,
            event.content,
            ts=instant_for(event.day, start=start),
        )
        candidate = candidate_for(event, start=start)
        if candidate is not None:
            policy.apply(store, candidate, logged)
    return store


def _run_query(
    router: QueryRouter, scenario: Scenario, query: ScenarioQuery, start: datetime
) -> QueryOutcome:
    actual = _answer(router, query, start)
    return QueryOutcome(
        scenario_id=scenario.scenario_id,
        kind=query.kind,
        subject=query.subject,
        predicate=query.predicate,
        expected=query.answer,
        actual=actual,
    )


def _answer(router: QueryRouter, query: ScenarioQuery, start: datetime) -> tuple[str, ...]:
    if query.kind is QueryKind.CURRENT:
        return router.current(query.subject, query.predicate).objects
    if query.kind is QueryKind.HISTORICAL:
        assert query.as_of_day is not None  # guaranteed by validate_all
        as_of = instant_for(query.as_of_day, start=start)
        return router.historical(query.subject, query.predicate, as_of).objects
    return router.evolution(query.subject, query.predicate).objects


def format_gate(reports: dict[str, SystemReport]) -> str:
    """Render the gate as a small fixed-width table (for the CLI runner)."""
    kinds = list(QueryKind)
    header = ["system", "overall", *[k.value for k in kinds]]
    rows = [header]
    for name, report in reports.items():
        by_kind = report.accuracy_by_kind()
        row = [name, f"{report.accuracy:.0%}"]
        row += [f"{by_kind[k]:.0%}" if k in by_kind else "—" for k in kinds]
        rows.append(row)

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    ]
    return "\n".join(lines)
