"""The query router answers current / historical / evolution over a slot."""

from __future__ import annotations

from mneme.domain.events import Actor, EventType
from mneme.domain.facts import ExtractedFact
from mneme.query.router import QueryRouter


def _insert(store, log, at, subject, predicate, obj, day):
    event = log.append(Actor.USER, EventType.MESSAGE, f"{subject}:{obj}", ts=at(day))
    return store.insert(
        ExtractedFact(subject=subject, predicate=predicate, object=obj, valid_from=at(day)),
        event.event_id,
        ingested_at=at(day),
    )


def _relocation_chain(store, log, at):
    """alice lives_in berlin (day 1) superseded by lisbon (day 28)."""
    berlin = _insert(store, log, at, "alice", "lives_in", "berlin", 1)
    lisbon = _insert(store, log, at, "alice", "lives_in", "lisbon", 28)
    store.close_out(berlin.fact_id, lisbon.fact_id, valid_to=at(28), superseded_at=at(28))
    return berlin, lisbon


# --- current ------------------------------------------------------------------


def test_current_returns_the_unsuperseded_belief(store, log, at):
    _relocation_chain(store, log, at)

    answer = QueryRouter(store).current("alice", "lives_in")

    assert answer.objects == ("lisbon",)
    assert tuple(f.object for f in answer.facts) == ("lisbon",)


def test_current_on_empty_slot_is_empty(store):
    answer = QueryRouter(store).current("nobody", "lives_in")

    assert answer.objects == ()
    assert answer.facts == ()


# --- historical ---------------------------------------------------------------


def test_historical_picks_belief_in_force_at_the_instant(store, log, at):
    _relocation_chain(store, log, at)
    router = QueryRouter(store)

    assert router.historical("alice", "lives_in", at(20)).objects == ("berlin",)
    assert router.historical("alice", "lives_in", at(1)).objects == ("berlin",)


def test_historical_at_closeout_instant_belongs_to_successor(store, log, at):
    # valid_to is exclusive: at day 28 berlin's interval has ended, lisbon's begun.
    _relocation_chain(store, log, at)

    answer = QueryRouter(store).historical("alice", "lives_in", at(28))

    assert answer.objects == ("lisbon",)


def test_historical_before_first_belief_is_empty(store, log, at):
    _relocation_chain(store, log, at)

    answer = QueryRouter(store).historical("alice", "lives_in", at(1, hour=0))

    # day 1 at 00:00 precedes valid_from (day 1 at 12:00).
    assert answer.objects == ()


# --- evolution ----------------------------------------------------------------


def test_evolution_walks_the_chain_oldest_first(store, log, at):
    _relocation_chain(store, log, at)

    answer = QueryRouter(store).evolution("alice", "lives_in")

    assert answer.objects == ("berlin", "lisbon")


def test_evolution_follows_a_three_step_chain(store, log, at):
    berlin = _insert(store, log, at, "alice", "lives_in", "berlin", 1)
    lisbon = _insert(store, log, at, "alice", "lives_in", "lisbon", 20)
    store.close_out(berlin.fact_id, lisbon.fact_id, valid_to=at(20), superseded_at=at(20))
    porto = _insert(store, log, at, "alice", "lives_in", "porto", 30)
    store.close_out(lisbon.fact_id, porto.fact_id, valid_to=at(30), superseded_at=at(30))

    answer = QueryRouter(store).evolution("alice", "lives_in")

    assert answer.objects == ("berlin", "lisbon", "porto")


def test_evolution_on_single_row_is_a_chain_of_one(store, log, at):
    # The B0 overwrite shape: one row per slot, history gone.
    _insert(store, log, at, "alice", "lives_in", "lisbon", 28)

    answer = QueryRouter(store).evolution("alice", "lives_in")

    assert answer.objects == ("lisbon",)


def test_evolution_on_empty_slot_is_empty(store):
    answer = QueryRouter(store).evolution("nobody", "lives_in")

    assert answer.objects == ()


# --- the B0 discriminator at the router level ---------------------------------


def test_overwrite_shape_loses_history_router_cannot_recover_it(store, log, at):
    # Simulate overwrite: only the latest belief exists, valid from day 28.
    _insert(store, log, at, "alice", "lives_in", "lisbon", 28)
    router = QueryRouter(store)

    # current still works...
    assert router.current("alice", "lives_in").objects == ("lisbon",)
    # ...but the past is unrecoverable: nothing was valid on day 20.
    assert router.historical("alice", "lives_in", at(20)).objects == ()
    # ...and evolution collapses to the single surviving belief.
    assert router.evolution("alice", "lives_in").objects == ("lisbon",)


# --- robustness: a corrupted chain falls back, never loops --------------------


def test_evolution_falls_back_when_slot_has_no_supersession_links(store, log, at):
    # Two current rows for one slot, never linked (the InsertOnly shape): no
    # unique head, so evolution falls back to slot order rather than guessing.
    _insert(store, log, at, "alice", "lives_in", "berlin", 1)
    _insert(store, log, at, "alice", "lives_in", "lisbon", 2)

    answer = QueryRouter(store).evolution("alice", "lives_in")

    assert answer.objects == ("berlin", "lisbon")


def test_evolution_falls_back_when_links_are_not_one_clean_chain(store, log, at, conn):
    # A healthy chain H -> A plus a separate two-row cycle C <-> D in the same
    # slot: one head but unreachable rows. The router must fall back to slot
    # order and terminate (the seen-guard prevents an infinite walk).
    head = _insert(store, log, at, "alice", "lives_in", "berlin", 1)
    a = _insert(store, log, at, "alice", "lives_in", "lisbon", 2)
    c = _insert(store, log, at, "alice", "lives_in", "porto", 3)
    d = _insert(store, log, at, "alice", "lives_in", "madrid", 4)
    conn.execute("UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (a.fact_id, head.fact_id))
    conn.execute("UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (d.fact_id, c.fact_id))
    conn.execute("UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (c.fact_id, d.fact_id))
    conn.commit()

    answer = QueryRouter(store).evolution("alice", "lives_in")

    # Fallback = slot order (valid_from, fact_id); all four rows, no crash.
    assert answer.objects == ("berlin", "lisbon", "porto", "madrid")
