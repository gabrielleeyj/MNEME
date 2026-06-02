"""Evaluation harness and synthetic dataset (workstream 7).

The dataset is the spec the project is judged by: known timelines whose facts,
supersession relations, and query answers are authored and self-checked, so the
B0 ablation (Supersede vs Overwrite) can be scored against ground truth.
"""

from mneme.eval.dataset import (
    DEFAULT_START,
    GOLD_SCENARIOS,
    candidate_for,
    instant_for,
    materialize,
    total_event_count,
)
from mneme.eval.harness import (
    OVERWRITE,
    SUPERSEDE,
    SYSTEMS,
    QueryOutcome,
    System,
    SystemReport,
    evaluate,
    format_gate,
    run_gate,
)
from mneme.eval.oracle import OracleError, ScenarioOracleDetector
from mneme.eval.scenario import (
    Assertion,
    QueryKind,
    Scenario,
    ScenarioEvent,
    ScenarioQuery,
)
from mneme.eval.validate import ScenarioError, validate_all, validate_scenario

__all__ = [
    "DEFAULT_START",
    "GOLD_SCENARIOS",
    "candidate_for",
    "instant_for",
    "materialize",
    "total_event_count",
    "Assertion",
    "QueryKind",
    "Scenario",
    "ScenarioEvent",
    "ScenarioQuery",
    "ScenarioError",
    "validate_all",
    "validate_scenario",
    "OracleError",
    "ScenarioOracleDetector",
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
