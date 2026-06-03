"""A procedural generator for the scale dataset — the real discriminator.

The hand-authored gold (``dataset.GOLD_SCENARIOS``) is tiny: ~4 events a
scenario, so top-k retrieval pulls the whole timeline and a capable LLM reasons
``historical``/``evolution`` out unaided. At that size the RAG-style baselines
(B1, B2) nearly tie supersession, and the structural advantage is invisible.

This module builds the thing that separates them: **one long timeline** (~1000
events by default) over many ``(subject, predicate)`` slots, each with a
multi-step belief history, buried in chatter. Length is the whole point — the
baselines ingest a scenario into a *single* store, so a long timeline is where
B1's fixed top-k can no longer recall the right past message and B2's running
summary has compressed superseded states away, while supersession answers from
structured rows regardless of length.

Ground truth is correct *by construction*: the generator knows each slot's
history as it builds it, so it emits the gold queries directly rather than
guessing them. ``validate_scenario`` is still run at the end as an independent
check. Generation is fully seeded, so a given ``GenConfig`` always yields the
same scenario — the dataset is reproducible and the tests are deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import count

from mneme.eval.scenario import (
    Assertion,
    QueryKind,
    Scenario,
    ScenarioEvent,
    ScenarioQuery,
)
from mneme.eval.validate import validate_scenario
from mneme.facts.detector import Relation

__all__ = ["GenConfig", "default_config", "scale_config", "generate"]


@dataclass(frozen=True, slots=True)
class _Predicate:
    """A predicate the generator can build a belief history for."""

    name: str
    noun: str  # for the evolution phrasing: "how has {subj}'s {noun} changed"
    values: tuple[str, ...]
    templates: tuple[str, ...]  # assertion content, "{subj} ... {obj}"
    current_q: str  # "Where does {subj} live now?"
    historical_q: str  # "Where did {subj} live on day {day}?"
    evolution_q: str  # "How has {subj}'s home changed over time?"
    refinable: bool = False


_PREDICATES: tuple[_Predicate, ...] = (
    _Predicate(
        name="lives_in",
        noun="home",
        values=(
            "berlin", "lisbon", "tokyo", "austin", "nairobi", "toronto",
            "madrid", "oslo", "seoul", "denver", "dublin", "lima",
        ),
        templates=(
            "{subj} just moved to {obj}.",
            "{subj} is settling into life in {obj}.",
            "{subj} relocated to {obj} this month.",
        ),
        current_q="Where does {subj} live now?",
        historical_q="Where did {subj} live on day {day}?",
        evolution_q="How has {subj}'s home changed over time?",
    ),
    _Predicate(
        name="works_at",
        noun="employer",
        values=(
            "acme", "globex", "initech", "umbrella", "hooli", "vandelay",
            "soylent", "stark", "wayne", "tyrell", "cyberdyne", "aperture",
        ),
        templates=(
            "{subj} started a new job at {obj}.",
            "{subj} signed with {obj}.",
            "{subj} is now working at {obj}.",
        ),
        current_q="Where does {subj} work now?",
        historical_q="Where did {subj} work on day {day}?",
        evolution_q="How has {subj}'s employer changed over time?",
        refinable=True,
    ),
    _Predicate(
        name="drives",
        noun="car",
        values=(
            "toyota", "honda", "volvo", "tesla", "subaru", "mazda",
            "ford", "kia", "audi", "fiat", "jeep", "lexus",
        ),
        templates=(
            "{subj} picked up a {obj}.",
            "{subj} is driving a {obj} these days.",
            "{subj} traded the old car for a {obj}.",
        ),
        current_q="What car does {subj} drive now?",
        historical_q="What car did {subj} drive on day {day}?",
        evolution_q="How has {subj}'s car changed over time?",
    ),
    _Predicate(
        name="role",
        noun="role",
        values=(
            "engineer", "designer", "analyst", "manager", "researcher",
            "writer", "teacher", "consultant", "nurse", "chef", "pilot", "lawyer",
        ),
        templates=(
            "{subj} took a role as a {obj}.",
            "{subj} is working as a {obj} now.",
            "{subj} switched careers to become a {obj}.",
        ),
        current_q="What is {subj}'s role now?",
        historical_q="What was {subj}'s role on day {day}?",
        evolution_q="How has {subj}'s role changed over time?",
        refinable=True,
    ),
    _Predicate(
        name="favourite_drink",
        noun="favourite drink",
        values=(
            "tea", "coffee", "matcha", "cocoa", "espresso", "chai",
            "kombucha", "lemonade", "cider", "horchata",
        ),
        templates=(
            "{subj} can't get enough {obj} lately.",
            "{subj} switched to {obj}.",
            "{subj} says {obj} is the new favourite.",
        ),
        current_q="What does {subj} like to drink now?",
        historical_q="What did {subj} drink on day {day}?",
        evolution_q="How has {subj}'s favourite drink changed over time?",
    ),
)

_NAMES: tuple[str, ...] = (
    "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi",
    "ivan", "judy", "mallory", "niaj", "olivia", "peggy", "rupert", "sybil",
    "trent", "uma", "victor", "wendy", "xander", "yara", "zach", "nina",
)

_REFINE_SUFFIXES: tuple[str, ...] = ("corp", "inc", "group", "ltd")

_CHATTER: tuple[str, ...] = (
    "{subj} had a relaxing weekend.",
    "{subj} is travelling for a conference this week.",
    "{subj} went for a long run this morning.",
    "{subj} is reading a great novel.",
    "{subj} caught up with old friends.",
    "{subj} tried a new recipe last night.",
    "{subj} is feeling under the weather.",
    "{subj} spent the afternoon at the museum.",
    "{subj} adopted a rescue dog.",
    "{subj} is learning to play the guitar.",
)


@dataclass(frozen=True, slots=True)
class GenConfig:
    """Knobs for the generator. Defaults aim at a ~1000-event timeline."""

    seed: int = 7
    num_subjects: int = 60
    predicates_per_subject: int = 3
    max_changes_per_slot: int = 4
    chatter_ratio: float = 0.5
    num_query_slots: int = 40
    scenario_id: str = "scale"


def default_config() -> GenConfig:
    """A small, fast config for tests — a few dozen events, fully deterministic."""
    return GenConfig(seed=1, num_subjects=4, num_query_slots=6, scenario_id="scale-small")


def scale_config() -> GenConfig:
    """The headline ~1000-event config used for the live discriminator run."""
    return GenConfig()


@dataclass(slots=True)
class _Slot:
    """A single subject+predicate belief history as it is generated."""

    subject: str
    predicate: _Predicate
    # Every event on the slot, including duplicate restatements (for content).
    events: list[tuple[int, str, Relation, str]] = field(default_factory=list)
    # Only the belief-*changing* steps (day, object) — the gold timeline.
    timeline: list[tuple[int, str]] = field(default_factory=list)

    @property
    def chain(self) -> tuple[str, ...]:
        return tuple(obj for _day, obj in self.timeline)


def generate(config: GenConfig | None = None) -> Scenario:
    """Build, validate, and return one long scenario from ``config``."""
    cfg = config if config is not None else scale_config()
    rng = random.Random(cfg.seed)

    slots = _build_slots(rng, cfg)
    events = _assemble_events(rng, cfg, slots)
    queries = _build_queries(rng, cfg, slots)

    scenario = Scenario(scenario_id=cfg.scenario_id, events=events, queries=queries)
    validate_scenario(scenario)  # independent check; correct by construction
    return scenario


def _build_slots(rng: random.Random, cfg: GenConfig) -> list[_Slot]:
    subjects = _subjects(cfg.num_subjects)
    span = _span(cfg)
    slots: list[_Slot] = []
    for subject in subjects:
        chosen = rng.sample(_PREDICATES, k=min(cfg.predicates_per_subject, len(_PREDICATES)))
        for predicate in chosen:
            slots.append(_build_slot(rng, subject, predicate, cfg, span))
    return slots


def _build_slot(
    rng: random.Random, subject: str, predicate: _Predicate, cfg: GenConfig, span: int
) -> _Slot:
    steps = _slot_steps(rng, predicate, cfg.max_changes_per_slot)
    days = sorted(rng.sample(range(span + 1), k=len(steps)))
    slot = _Slot(subject=subject, predicate=predicate)
    for day, (relation, obj) in zip(days, steps):
        content = _render(rng, predicate, subject, obj)
        slot.events.append((day, content, relation, obj))
        if relation is not Relation.DUPLICATE:
            slot.timeline.append((day, obj))
    return slot


def _slot_steps(
    rng: random.Random, predicate: _Predicate, max_changes: int
) -> list[tuple[Relation, str]]:
    pool = list(predicate.values)
    rng.shuffle(pool)
    current = pool.pop()
    steps: list[tuple[Relation, str]] = [(Relation.NEW, current)]

    for _ in range(rng.randint(0, max_changes)):
        roll = rng.random()
        if roll < 0.2:
            steps.append((Relation.DUPLICATE, current))  # restate, no change
        elif roll < 0.4 and predicate.refinable:
            current = f"{current} {rng.choice(_REFINE_SUFFIXES)}"
            steps.append((Relation.REFINES, current))
        elif pool:
            current = pool.pop()
            steps.append((Relation.SUPERSEDES, current))
    return steps


def _assemble_events(
    rng: random.Random, cfg: GenConfig, slots: list[_Slot]
) -> tuple[ScenarioEvent, ...]:
    span = _span(cfg)
    dated: list[tuple[int, int, ScenarioEvent]] = []
    tiebreak = count()

    for slot in slots:
        for day, content, relation, obj in slot.events:
            assertion = Assertion(slot.subject, slot.predicate.name, obj, relation)
            dated.append((day, next(tiebreak), ScenarioEvent(day, content, assertion)))

    for day, content in _chatter(rng, cfg, slots, span):
        dated.append((day, next(tiebreak), ScenarioEvent(day, content)))

    dated.sort(key=lambda item: (item[0], item[1]))
    return tuple(event for _day, _seq, event in dated)


def _chatter(
    rng: random.Random, cfg: GenConfig, slots: list[_Slot], span: int
) -> list[tuple[int, str]]:
    assertion_count = sum(len(slot.events) for slot in slots)
    ratio = min(max(cfg.chatter_ratio, 0.0), 0.95)
    chatter_count = round(assertion_count * ratio / (1.0 - ratio))
    subjects = [slot.subject for slot in slots] or list(_NAMES)
    out: list[tuple[int, str]] = []
    for _ in range(chatter_count):
        subject = rng.choice(subjects)
        template = rng.choice(_CHATTER)
        out.append((rng.randint(0, span), template.format(subj=subject)))
    return out


def _build_queries(
    rng: random.Random, cfg: GenConfig, slots: list[_Slot]
) -> tuple[ScenarioQuery, ...]:
    rich = [slot for slot in slots if len(slot.timeline) >= 2]
    flat = [slot for slot in slots if len(slot.timeline) == 1]
    rng.shuffle(rich)
    rng.shuffle(flat)

    queries: list[ScenarioQuery] = []
    for slot in rich[: cfg.num_query_slots]:
        queries.append(_current_query(slot))
        queries.append(_evolution_query(slot))
        queries.append(_historical_query(rng, slot))

    # A few single-change slots as controls: current is all they can answer.
    for slot in flat[: max(1, cfg.num_query_slots // 4)]:
        queries.append(_current_query(slot))
    return tuple(queries)


def _current_query(slot: _Slot) -> ScenarioQuery:
    return ScenarioQuery(
        kind=QueryKind.CURRENT,
        subject=slot.subject,
        predicate=slot.predicate.name,
        question=slot.predicate.current_q.format(subj=slot.subject),
        answer=(slot.timeline[-1][1],),
    )


def _evolution_query(slot: _Slot) -> ScenarioQuery:
    return ScenarioQuery(
        kind=QueryKind.EVOLUTION,
        subject=slot.subject,
        predicate=slot.predicate.name,
        question=slot.predicate.evolution_q.format(subj=slot.subject),
        answer=slot.chain,
    )


def _historical_query(rng: random.Random, slot: _Slot) -> ScenarioQuery:
    # Aim at a superseded interval [d_i, d_{i+1}) so only a history-preserving
    # store can answer it. Days are distinct, so the interval is non-empty.
    i = rng.randrange(len(slot.timeline) - 1)
    day_i, value_i = slot.timeline[i]
    day_next = slot.timeline[i + 1][0]
    as_of = rng.randint(day_i, day_next - 1)
    return ScenarioQuery(
        kind=QueryKind.HISTORICAL,
        subject=slot.subject,
        predicate=slot.predicate.name,
        question=slot.predicate.historical_q.format(subj=slot.subject, day=as_of),
        answer=(value_i,),
        as_of_day=as_of,
    )


def _render(rng: random.Random, predicate: _Predicate, subject: str, obj: str) -> str:
    return rng.choice(predicate.templates).format(subj=subject, obj=obj)


def _subjects(num_subjects: int) -> list[str]:
    out = list(_NAMES[:num_subjects])
    suffix = 2
    while len(out) < num_subjects:
        for name in _NAMES:
            if len(out) >= num_subjects:
                break
            out.append(f"{name}{suffix}")
        suffix += 1
    return out


def _span(cfg: GenConfig) -> int:
    # Enough distinct days that every slot's steps get strictly increasing days.
    return max(cfg.num_subjects * cfg.max_changes_per_slot * 3, 64)
