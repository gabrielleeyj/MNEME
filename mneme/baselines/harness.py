"""Score the RAG-style baselines (B1 raw RAG, B2 summary) over the gold.

For each scenario this ingests *every* message (fact-bearing and chatter alike)
into a fresh baseline, asks each gold question, and grades the free-text answer
with the LLM judge. The result mirrors the B0 gate's rollups — overall and per
query kind — but on the baselines' own scoring path: NL answer + LLM judge, not
exact match on structured objects. The two paths are reported side by side, not
merged, so each stays honest about how it was graded.

The embedder (B1 only) and the LLM client are injected: pass real ones
(FastEmbed + Anthropic) for the live number, or fakes for an offline,
deterministic test.
"""

from __future__ import annotations

from dataclasses import dataclass

from mneme.baselines.judge import LLMJudge
from mneme.baselines.rag import DEFAULT_TOP_K, RawRagBaseline
from mneme.baselines.summary import SummaryBaseline
from mneme.embeddings.client import EmbeddingClient
from mneme.eval.dataset import GOLD_SCENARIOS
from mneme.eval.scenario import QueryKind, Scenario, ScenarioQuery
from mneme.eval.validate import validate_all
from mneme.index.semantic_index import SemanticIndex
from mneme.llm.client import LLMClient

__all__ = [
    "B1_RAW_RAG",
    "B2_SUMMARY",
    "BaselineOutcome",
    "BaselineReport",
    "evaluate_rag",
    "evaluate_summary",
    "format_baselines",
]

# The names these baselines are reported under, alongside the B0 gate's systems.
B1_RAW_RAG = "b1-raw-rag"
B2_SUMMARY = "b2-summary"


@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    """One gold query under a baseline: the answer and the judge's verdict."""

    scenario_id: str
    kind: QueryKind
    question: str
    reference: tuple[str, ...]
    answer: str
    correct: bool
    reason: str


@dataclass(frozen=True, slots=True)
class BaselineReport:
    """Every outcome for one baseline, with accuracy rollups."""

    system: str
    outcomes: tuple[BaselineOutcome, ...]

    @property
    def accuracy(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.correct for o in self.outcomes) / len(self.outcomes)

    def accuracy_by_kind(self) -> dict[QueryKind, float]:
        scores: dict[QueryKind, float] = {}
        for kind in QueryKind:
            of_kind = [o for o in self.outcomes if o.kind is kind]
            if of_kind:
                scores[kind] = sum(o.correct for o in of_kind) / len(of_kind)
        return scores


def evaluate_rag(
    embedder: EmbeddingClient,
    client: LLMClient,
    scenarios: tuple[Scenario, ...] = GOLD_SCENARIOS,
    *,
    judge_client: LLMClient | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> BaselineReport:
    """Score raw RAG across the scenarios. Validates the gold first.

    ``judge_client`` defaults to ``client`` — one model answers and grades, as in
    the live single-key run.
    """
    validate_all(scenarios)
    judge = LLMJudge(judge_client if judge_client is not None else client)
    outcomes = tuple(
        outcome
        for scenario in scenarios
        for outcome in _evaluate_scenario(embedder, client, judge, scenario, top_k)
    )
    return BaselineReport(B1_RAW_RAG, outcomes)


def evaluate_summary(
    client: LLMClient,
    scenarios: tuple[Scenario, ...] = GOLD_SCENARIOS,
    *,
    judge_client: LLMClient | None = None,
) -> BaselineReport:
    """Score the running-summary baseline across the scenarios.

    No embedder: B2 keeps no raw messages to retrieve. ``judge_client`` defaults
    to ``client``, as in the live single-key run.
    """
    validate_all(scenarios)
    judge = LLMJudge(judge_client if judge_client is not None else client)
    outcomes = tuple(
        outcome
        for scenario in scenarios
        for outcome in _evaluate_summary_scenario(client, judge, scenario)
    )
    return BaselineReport(B2_SUMMARY, outcomes)


def _evaluate_scenario(
    embedder: EmbeddingClient,
    client: LLMClient,
    judge: LLMJudge,
    scenario: Scenario,
    top_k: int,
) -> list[BaselineOutcome]:
    baseline = RawRagBaseline(SemanticIndex(embedder), client, top_k=top_k)
    for offset, event in enumerate(scenario.events, start=1):
        baseline.ingest(offset, event.content, display=f"[day {event.day}] {event.content}")
    return [_run_query(baseline, judge, scenario, query) for query in scenario.queries]


def _evaluate_summary_scenario(
    client: LLMClient,
    judge: LLMJudge,
    scenario: Scenario,
) -> list[BaselineOutcome]:
    baseline = SummaryBaseline(client)
    for event in scenario.events:
        baseline.ingest(f"[day {event.day}] {event.content}")
    return [_run_query(baseline, judge, scenario, query) for query in scenario.queries]


class _Answerer:
    """Anything that answers a question from its ingested state (B1/B2)."""

    def answer(self, question: str) -> str: ...  # pragma: no cover - typing only


def _run_query(
    baseline: _Answerer,
    judge: LLMJudge,
    scenario: Scenario,
    query: ScenarioQuery,
) -> BaselineOutcome:
    answer = baseline.answer(query.question)
    verdict = judge.judge(query.question, query.answer, answer)
    return BaselineOutcome(
        scenario_id=scenario.scenario_id,
        kind=query.kind,
        question=query.question,
        reference=query.answer,
        answer=answer,
        correct=verdict.correct,
        reason=verdict.reason,
    )


def format_baselines(reports: dict[str, BaselineReport]) -> str:
    """Render baselines as a fixed-width table (overall + per query kind)."""
    kinds = list(QueryKind)
    header = ["system", "overall", *[k.value for k in kinds]]
    rows = [header]
    for name, report in reports.items():
        by_kind = report.accuracy_by_kind()
        row = [name, f"{report.accuracy:.0%}"]
        row += [f"{by_kind[k]:.0%}" if k in by_kind else "—" for k in kinds]
        rows.append(row)

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows
    )
