#!/usr/bin/env python3
"""Run the B0 gate offline and print the supersede / overwrite / B3 table.

No API key needed: the harness swaps the LLM judge for the gold oracle, so the
only variable between supersede and overwrite is the storage policy. B3 (a
Graphiti-like bitemporal store) is driven by the same gold relations and scored
on the same structured exact-match path, so the only variable between supersede
and B3 is the *substrate*. Two gaps matter:

  * supersede vs overwrite on `historical`/`evolution` — the B0 result: is
    supersession worth it at all? (Yes.)
  * supersede vs B3 — a *tie* keeps the substrate question open: at this scale an
    event-sourced projection and a direct bitemporal graph answer identically.

    python scripts/eval_harness.py
"""

from __future__ import annotations

from mneme.baselines.bitemporal import B3_BITEMPORAL, evaluate_bitemporal
from mneme.eval.harness import format_gate, run_gate


def main() -> None:
    reports = run_gate()
    reports[B3_BITEMPORAL] = evaluate_bitemporal()
    print(format_gate(reports))


if __name__ == "__main__":
    main()
