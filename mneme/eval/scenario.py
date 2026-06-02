"""The synthetic-dataset model — the spec the whole project scores against.

A ``Scenario`` is a known timeline: a sequence of events whose ground-truth
facts and supersession relations are authored by hand, plus queries with gold
answers. It serves two masters at once:

  * It is the *spec for the WS3 detector*. Each fact-bearing event carries the
    relation it should be judged as (new / duplicate / refines / supersedes),
    so the detector's false-supersession rate can be scored directly.
  * It is the *discriminator for the B0 gate*. ``historical`` and ``evolution``
    queries have answers that only a history-preserving store can produce, so
    Supersede and Overwrite (B0) separate measurably.

The relation vocabulary is reused from the detector (``mneme.facts.detector``)
so the gold label and the system's verdict share one type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mneme.facts.detector import Relation

__all__ = [
    "QueryKind",
    "Assertion",
    "ScenarioEvent",
    "ScenarioQuery",
    "Scenario",
]


class QueryKind(str, Enum):
    """What a query asks of memory.

    ``CURRENT`` and ``HISTORICAL`` and ``EVOLUTION`` map to the three retrieval
    modes the query router (WS5) must serve. Only ``CURRENT`` is answerable by a
    store that overwrites; the other two need the preserved chain.
    """

    CURRENT = "current"
    HISTORICAL = "historical"
    EVOLUTION = "evolution"


@dataclass(frozen=True, slots=True)
class Assertion:
    """The ground-truth fact an event asserts, plus how it relates to prior state.

    ``relation`` is the gold label for the detector: it is the relation this
    assertion bears to the *current* belief on its ``(subject, predicate)`` slot
    at the moment the event arrives.
    """

    subject: str
    predicate: str
    object: str
    relation: Relation


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """One message in a timeline. ``day`` is an offset from the scenario start.

    ``assertion`` is ``None`` for chatter that asserts no fact — those events
    test that the extractor does *not* invent facts, and must never change any
    slot's belief.
    """

    day: int
    content: str
    assertion: Assertion | None = None


@dataclass(frozen=True, slots=True)
class ScenarioQuery:
    """A question with a gold answer, anchored to a specific belief slot.

    ``answer`` is a tuple so evolution queries can carry the ordered chain of
    objects; current/historical answers are single-element. ``as_of_day`` is
    required for ``HISTORICAL`` (the point in valid-time being asked about) and
    must be ``None`` for the other kinds.
    """

    kind: QueryKind
    subject: str
    predicate: str
    question: str
    answer: tuple[str, ...]
    as_of_day: int | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named timeline plus its gold queries."""

    scenario_id: str
    events: tuple[ScenarioEvent, ...]
    queries: tuple[ScenarioQuery, ...]
