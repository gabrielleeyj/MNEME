"""Evaluation harness and synthetic dataset (workstream 7).

The dataset is the spec the project is judged by: known timelines whose facts,
supersession relations, and query answers are authored and self-checked, so the
B0 ablation (Supersede vs Overwrite) can be scored against ground truth.
"""

from mneme.eval.dataset import (
    DEFAULT_START,
    GOLD_SCENARIOS,
    materialize,
    total_event_count,
)
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
]
