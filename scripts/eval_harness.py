#!/usr/bin/env python3
"""Run the B0 gate offline and print the supersede-vs-overwrite table.

No API key needed: the harness swaps the LLM judge for the gold oracle, so the
only variable between the two systems is the storage policy. The number that
matters is the gap on the `historical` and `evolution` columns.

    python scripts/eval_harness.py
"""

from __future__ import annotations

from mneme.eval.harness import format_gate, run_gate


def main() -> None:
    reports = run_gate()
    print(format_gate(reports))


if __name__ == "__main__":
    main()
