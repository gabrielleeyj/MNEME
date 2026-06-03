#!/usr/bin/env python3
"""Generate the ~1000-event scale dataset and run the offline gate over it.

No API key needed: the generator is fully procedural and the gate swaps the LLM
judge for the gold oracle, so the only variable between supersede and overwrite
is the storage policy, and B3 (a Graphiti-like bitemporal store) is driven by the
same gold relations. This is the *discriminator* the toy gold cannot show: at
length, overwrite collapses on ``historical``/``evolution`` while supersede and
B3 still answer from preserved history.

    python scripts/generate_dataset.py
"""

from __future__ import annotations

from collections import Counter

from mneme.baselines.bitemporal import B3_BITEMPORAL, evaluate_bitemporal
from mneme.eval.generator import generate, scale_config
from mneme.eval.harness import format_gate, run_gate
from mneme.eval.scenario import Scenario


def _print_stats(scenario: Scenario) -> None:
    facts = [event for event in scenario.events if event.assertion is not None]
    chatter = [event for event in scenario.events if event.assertion is None]
    kinds = Counter(query.kind.value for query in scenario.queries)

    print(f"scenario:  {scenario.scenario_id}")
    print(f"events:    {len(scenario.events)} ({len(facts)} facts, {len(chatter)} chatter)")
    print(f"queries:   {len(scenario.queries)} {dict(kinds)}")
    print()


def main() -> None:
    scenario = generate(scale_config())
    _print_stats(scenario)

    reports = run_gate(scenarios=(scenario,))
    reports[B3_BITEMPORAL] = evaluate_bitemporal(scenarios=(scenario,))
    print(format_gate(reports))


if __name__ == "__main__":
    main()
