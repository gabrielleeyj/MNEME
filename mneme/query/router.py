"""The query router — three ways to ask memory a question (workstream 5).

A question always names a belief slot (``subject`` + ``predicate``); the router
answers it one of three ways:

  * ``current``    — what is believed now (the un-superseded row).
  * ``historical`` — what was believed at a point in valid-time (the row whose
    validity interval covers it).
  * ``evolution``  — how the belief changed, by walking the ``superseded_by``
    chain oldest-first.

Only ``current`` survives the B0 overwrite ablation: overwrite keeps one row per
slot, so ``historical`` for a past instant finds nothing and ``evolution`` is a
chain of one. That gap is exactly what the eval harness measures, so the router
stays deterministic and LLM-free — resolving a natural-language question to a
slot is not its job.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from mneme.domain.facts import Fact
from mneme.facts.store import FactStore

__all__ = ["Answer", "QueryRouter"]


@dataclass(frozen=True, slots=True)
class Answer:
    """A router result: the objects answering the question plus their facts.

    ``objects`` holds one value for ``current``/``historical`` (empty when the
    slot has no belief at all, or none in force at the asked instant) and the
    ordered chain for ``evolution``. ``facts`` are the supporting rows, aligned
    with ``objects``.
    """

    subject: str
    predicate: str
    objects: tuple[str, ...]
    facts: tuple[Fact, ...]


class QueryRouter:
    def __init__(self, store: FactStore) -> None:
        self._store = store

    def current(self, subject: str, predicate: str) -> Answer:
        """The belief in force now for ``subject``/``predicate``."""
        fact = self._store.current_for(subject, predicate)
        facts = (fact,) if fact is not None else ()
        return _answer(subject, predicate, facts)

    def historical(self, subject: str, predicate: str, as_of: datetime) -> Answer:
        """The belief in force at ``as_of`` — the row whose interval covers it."""
        match = _fact_in_force_at(self._store.slot_facts(subject, predicate), as_of)
        facts = (match,) if match is not None else ()
        return _answer(subject, predicate, facts)

    def evolution(self, subject: str, predicate: str) -> Answer:
        """The ordered history of the slot, following ``superseded_by``."""
        chain = _walk_chain(self._store.slot_facts(subject, predicate))
        return _answer(subject, predicate, chain)


def _answer(subject: str, predicate: str, facts: tuple[Fact, ...]) -> Answer:
    return Answer(
        subject=subject,
        predicate=predicate,
        objects=tuple(fact.object for fact in facts),
        facts=facts,
    )


def _fact_in_force_at(facts: list[Fact], as_of: datetime) -> Fact | None:
    """The fact whose validity interval ``[valid_from, valid_to)`` contains ``as_of``.

    Scans oldest-first and returns the first cover, so a clean chain yields the
    unique belief in force; ``valid_to is None`` means still in force.
    """
    for fact in facts:
        if fact.valid_from <= as_of and (fact.valid_to is None or as_of < fact.valid_to):
            return fact
    return None


def _walk_chain(facts: list[Fact]) -> tuple[Fact, ...]:
    """Order a slot's facts by following ``superseded_by`` from its head.

    The head is the row no other row supersedes into. Walking the links (rather
    than trusting timestamps) reconstructs the exact succession the Supersede
    policy recorded; a slot with one row is a chain of one. Falls back to the
    given order if the links don't form a single clean chain.
    """
    if len(facts) <= 1:
        return tuple(facts)

    by_id = {fact.fact_id: fact for fact in facts}
    successor_ids = {
        fact.superseded_by for fact in facts if fact.superseded_by is not None
    }
    heads = [fact for fact in facts if fact.fact_id not in successor_ids]
    if len(heads) != 1:
        return tuple(facts)

    chain: list[Fact] = []
    seen: set[int] = set()
    node: Fact | None = heads[0]
    while node is not None and node.fact_id not in seen:
        chain.append(node)
        seen.add(node.fact_id)
        node = by_id.get(node.superseded_by) if node.superseded_by is not None else None

    if len(chain) != len(facts):
        return tuple(facts)
    return tuple(chain)
