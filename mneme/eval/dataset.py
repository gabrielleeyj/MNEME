"""The curated gold scenarios and their materialization into events.

These hand-authored timelines are the seed of WS7's synthetic dataset: small,
readable, and internally checked by ``validate.validate_scenario`` so they can
be trusted as ground truth. They deliberately exercise every relation the
detector must distinguish and every query mode that separates Supersede from
the B0 overwrite baseline.

``materialize`` turns a scenario into a tuple of immutable ``Event`` objects
with deterministic ids and timestamps; the assertion gold for ``events[i]`` is
``scenario.events[i].assertion``, so a harness zips the two together.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mneme.domain.events import Actor, Event, EventType
from mneme.domain.facts import ExtractedFact
from mneme.eval.scenario import (
    Assertion,
    QueryKind,
    Scenario,
    ScenarioEvent,
    ScenarioQuery,
)
from mneme.facts.detector import Relation

__all__ = [
    "DEFAULT_START",
    "GOLD_SCENARIOS",
    "candidate_for",
    "instant_for",
    "materialize",
    "total_event_count",
]

# All scenario days are offsets from this instant, in UTC.
DEFAULT_START = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# A relocation: one clean supersession plus chatter and a restatement.
_RELOCATION = Scenario(
    scenario_id="alice-relocation",
    events=(
        ScenarioEvent(
            day=0,
            content="alice:berlin",
            assertion=Assertion("alice", "lives_in", "berlin", Relation.NEW),
        ),
        ScenarioEvent(day=3, content="alice loves the coffee here"),
        ScenarioEvent(
            day=10,
            content="alice:berlin",
            assertion=Assertion("alice", "lives_in", "berlin", Relation.DUPLICATE),
        ),
        ScenarioEvent(
            day=40,
            content="alice:lisbon",
            assertion=Assertion("alice", "lives_in", "lisbon", Relation.SUPERSEDES),
        ),
    ),
    queries=(
        ScenarioQuery(
            kind=QueryKind.CURRENT,
            subject="alice",
            predicate="lives_in",
            question="Where does alice live now?",
            answer=("lisbon",),
        ),
        ScenarioQuery(
            kind=QueryKind.HISTORICAL,
            subject="alice",
            predicate="lives_in",
            question="Where did alice live on day 20?",
            answer=("berlin",),
            as_of_day=20,
        ),
        ScenarioQuery(
            kind=QueryKind.EVOLUTION,
            subject="alice",
            predicate="lives_in",
            question="How has alice's home changed over time?",
            answer=("berlin", "lisbon"),
        ),
    ),
)


# A career: a refine (vaguer -> sharper) then a real job change (supersede).
_CAREER = Scenario(
    scenario_id="bob-career",
    events=(
        ScenarioEvent(
            day=0,
            content="bob:acme",
            assertion=Assertion("bob", "works_at", "acme", Relation.NEW),
        ),
        ScenarioEvent(
            day=5,
            content="bob:acme corp",
            assertion=Assertion("bob", "works_at", "acme corp", Relation.REFINES),
        ),
        ScenarioEvent(day=12, content="bob is travelling for a conference"),
        ScenarioEvent(
            day=30,
            content="bob:globex",
            assertion=Assertion("bob", "works_at", "globex", Relation.SUPERSEDES),
        ),
    ),
    queries=(
        ScenarioQuery(
            kind=QueryKind.CURRENT,
            subject="bob",
            predicate="works_at",
            question="Where does bob work now?",
            answer=("globex",),
        ),
        ScenarioQuery(
            kind=QueryKind.HISTORICAL,
            subject="bob",
            predicate="works_at",
            question="Where did bob work on day 10?",
            answer=("acme corp",),
            as_of_day=10,
        ),
        ScenarioQuery(
            kind=QueryKind.EVOLUTION,
            subject="bob",
            predicate="works_at",
            question="How has bob's employer changed over time?",
            answer=("acme", "acme corp", "globex"),
        ),
    ),
)


# A stable preference: asserted once, never changed. Evolution is a chain of one,
# so a history-preserving store and an overwrite store must agree here — the
# control that keeps the discriminator honest.
_PREFERENCE = Scenario(
    scenario_id="alice-preference",
    events=(
        ScenarioEvent(
            day=2,
            content="alice likes tea",
            assertion=Assertion("alice", "likes", "tea", Relation.NEW),
        ),
        ScenarioEvent(
            day=15,
            content="alice likes tea",
            assertion=Assertion("alice", "likes", "tea", Relation.DUPLICATE),
        ),
    ),
    queries=(
        ScenarioQuery(
            kind=QueryKind.CURRENT,
            subject="alice",
            predicate="likes",
            question="What does alice like?",
            answer=("tea",),
        ),
        ScenarioQuery(
            kind=QueryKind.EVOLUTION,
            subject="alice",
            predicate="likes",
            question="How has alice's preference changed?",
            answer=("tea",),
        ),
    ),
)


GOLD_SCENARIOS: tuple[Scenario, ...] = (_RELOCATION, _CAREER, _PREFERENCE)


def materialize(
    scenario: Scenario,
    *,
    start: datetime = DEFAULT_START,
    first_event_id: int = 1,
) -> tuple[Event, ...]:
    """Render a scenario's events as immutable log events.

    Event ``i`` corresponds to ``scenario.events[i]``; ids run consecutively
    from ``first_event_id`` and timestamps are ``start`` plus the event's day.
    """
    events: list[Event] = []
    for offset, scenario_event in enumerate(scenario.events):
        events.append(
            Event(
                event_id=first_event_id + offset,
                ts=start + timedelta(days=scenario_event.day),
                actor=Actor.USER,
                type=EventType.MESSAGE,
                content=scenario_event.content,
            )
        )
    return tuple(events)


def instant_for(day: int, *, start: datetime = DEFAULT_START) -> datetime:
    """The absolute timestamp a scenario day offset maps to."""
    return start + timedelta(days=day)


def candidate_for(
    event: ScenarioEvent, *, start: datetime = DEFAULT_START
) -> ExtractedFact | None:
    """The fact candidate an event asserts, or ``None`` for chatter.

    The harness and the oracle both build candidates through this one helper, so
    the candidate the policy stores is byte-for-byte the key the oracle looks up
    its gold relation under.
    """
    if event.assertion is None:
        return None
    assertion = event.assertion
    return ExtractedFact(
        subject=assertion.subject,
        predicate=assertion.predicate,
        object=assertion.object,
        valid_from=instant_for(event.day, start=start),
    )


def total_event_count(scenarios: tuple[Scenario, ...] = GOLD_SCENARIOS) -> int:
    """Total number of events across the given scenarios."""
    return sum(len(scenario.events) for scenario in scenarios)
