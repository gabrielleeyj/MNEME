"""Scenario validation — proves the gold labels are internally consistent.

A hand-authored dataset is only a spec if it cannot quietly contradict itself.
This module replays a scenario's events the way the store would, tracking the
current belief per ``(subject, predicate)`` slot and the order in which beliefs
changed, then checks two things:

  * every assertion's ``relation`` is consistent with the slot's state at that
    point (a ``SUPERSEDES`` must actually contradict a prior belief; a
    ``DUPLICATE`` must restate it exactly; a ``NEW`` must have nothing to
    replace), and
  * every query's gold answer matches the belief history the events imply.

A scenario that passes is a usable spec; one that fails raises ``ScenarioError``
naming the offending event or query, so dataset bugs surface at authoring time
rather than as mysterious eval results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mneme.eval.scenario import QueryKind, Scenario, ScenarioQuery
from mneme.facts.detector import Relation

__all__ = ["ScenarioError", "validate_scenario", "validate_all"]


class ScenarioError(ValueError):
    """A scenario's gold labels or answers are internally inconsistent."""


@dataclass(slots=True)
class _Slot:
    """The replayed belief history for one (subject, predicate) pair."""

    current: str | None = None
    # (day, object) for each assertion that created or changed the belief.
    timeline: list[tuple[int, str]] = field(default_factory=list)

    def object_as_of(self, day: int) -> str | None:
        """The belief in force on ``day`` (the latest change at or before it)."""
        held: str | None = None
        for changed_on, obj in self.timeline:
            if changed_on <= day:
                held = obj
            else:
                break
        return held

    @property
    def chain(self) -> tuple[str, ...]:
        """The ordered objects this slot has believed, oldest first."""
        return tuple(obj for _day, obj in self.timeline)


def validate_scenario(scenario: Scenario) -> None:
    """Raise ``ScenarioError`` unless the scenario is internally consistent."""
    slots = _replay_events(scenario)
    for query in scenario.queries:
        _check_query(scenario.scenario_id, query, slots)


def validate_all(scenarios: tuple[Scenario, ...]) -> None:
    """Validate every scenario and reject duplicate ids."""
    seen: set[str] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen:
            raise ScenarioError(f"duplicate scenario_id {scenario.scenario_id!r}")
        seen.add(scenario.scenario_id)
        validate_scenario(scenario)


def _replay_events(scenario: Scenario) -> dict[tuple[str, str], _Slot]:
    slots: dict[tuple[str, str], _Slot] = {}
    previous_day: int | None = None

    for index, event in enumerate(scenario.events):
        if previous_day is not None and event.day < previous_day:
            raise ScenarioError(
                f"{scenario.scenario_id}: event {index} day {event.day} "
                f"precedes earlier event day {previous_day}"
            )
        previous_day = event.day

        if event.assertion is None:
            continue

        assertion = event.assertion
        key = (assertion.subject, assertion.predicate)
        slot = slots.setdefault(key, _Slot())
        _apply_assertion(scenario.scenario_id, index, assertion, slot, event.day)

    return slots


def _apply_assertion(
    scenario_id: str,
    index: int,
    assertion,
    slot: _Slot,
    day: int,
) -> None:
    relation = assertion.relation
    where = f"{scenario_id}: event {index} ({assertion.subject}/{assertion.predicate})"

    if relation is Relation.NEW:
        if slot.current is not None:
            raise ScenarioError(
                f"{where} is NEW but slot already believes {slot.current!r}"
            )
        slot.current = assertion.object
        slot.timeline.append((day, assertion.object))
        return

    if slot.current is None:
        raise ScenarioError(f"{where} is {relation.value} but slot has no prior belief")

    if relation is Relation.DUPLICATE:
        if assertion.object != slot.current:
            raise ScenarioError(
                f"{where} is DUPLICATE but {assertion.object!r} != {slot.current!r}"
            )
        return

    # REFINES and SUPERSEDES both replace the belief and must actually change it.
    if assertion.object == slot.current:
        raise ScenarioError(
            f"{where} is {relation.value} but does not change belief {slot.current!r}"
        )
    slot.current = assertion.object
    slot.timeline.append((day, assertion.object))


def _check_query(
    scenario_id: str,
    query: ScenarioQuery,
    slots: dict[tuple[str, str], _Slot],
) -> None:
    where = f"{scenario_id}: query {query.kind.value} ({query.subject}/{query.predicate})"

    if not query.answer:
        raise ScenarioError(f"{where} has an empty gold answer")

    slot = slots.get((query.subject, query.predicate))
    if slot is None or not slot.timeline:
        raise ScenarioError(f"{where} targets a slot no event ever asserts")

    if query.kind is QueryKind.HISTORICAL:
        _check_historical(where, query, slot)
        return

    if query.as_of_day is not None:
        raise ScenarioError(f"{where} is {query.kind.value} but sets as_of_day")

    if query.kind is QueryKind.CURRENT:
        expected = (slot.current,)
    else:  # EVOLUTION
        expected = slot.chain

    if query.answer != expected:
        raise ScenarioError(
            f"{where} gold answer {query.answer} != derived {expected}"
        )


def _check_historical(where: str, query: ScenarioQuery, slot: _Slot) -> None:
    if query.as_of_day is None:
        raise ScenarioError(f"{where} is HISTORICAL but has no as_of_day")
    held = slot.object_as_of(query.as_of_day)
    if held is None:
        raise ScenarioError(
            f"{where} as_of_day {query.as_of_day} precedes the first belief"
        )
    if query.answer != (held,):
        raise ScenarioError(
            f"{where} gold answer {query.answer} != belief {held!r} "
            f"as of day {query.as_of_day}"
        )
