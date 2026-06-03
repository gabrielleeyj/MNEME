"""B2 running summary: incremental folding, answer-from-summary, scored rollups.

All offline: a routing fake LLM stands in for the summary updater, the answerer,
and the judge so the whole harness runs deterministically with no network.
"""

from __future__ import annotations

import json

import pytest

from mneme.baselines.harness import B2_SUMMARY, evaluate_summary, format_baselines
from mneme.baselines.prompts import (
    JUDGE_SYSTEM_PROMPT,
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    SUMMARY_UPDATE_SYSTEM_PROMPT,
)
from mneme.baselines.summary import SummaryBaseline
from mneme.eval.dataset import GOLD_SCENARIOS
from mneme.eval.scenario import QueryKind


class CannedLLM:
    """Returns one fixed completion and records every call. No network."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


class RoutingFakeLLM:
    """Updates on the update prompt, answers on the answer prompt, grades on judge.

    ``correct_fn`` decides each judge verdict from the judge's user prompt, so a
    test can make whole query kinds pass or fail on demand.
    """

    def __init__(self, *, answer="an answer", correct_fn=None) -> None:
        self.answer = answer
        self.correct_fn = correct_fn or (lambda user: True)
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user})
        if system == JUDGE_SYSTEM_PROMPT:
            return json.dumps({"correct": self.correct_fn(user), "reason": "r"})
        if system == SUMMARY_UPDATE_SYSTEM_PROMPT:
            return "a running summary"
        return self.answer


# --- the running summary (B2) -------------------------------------------------


def test_ingest_folds_each_message_into_the_running_summary():
    llm = CannedLLM("alice lives in berlin")
    baseline = SummaryBaseline(llm)

    baseline.ingest("[day 0] alice:berlin")

    assert baseline.summary == "alice lives in berlin"
    (call,) = llm.calls
    assert call["system"] == SUMMARY_UPDATE_SYSTEM_PROMPT
    # The new message and the (empty) prior summary are both fed in.
    assert "[day 0] alice:berlin" in call["user"]


def test_ingest_passes_the_prior_summary_into_the_next_update():
    llm = CannedLLM("summary text")
    baseline = SummaryBaseline(llm)

    baseline.ingest("[day 0] alice:berlin")
    baseline.ingest("[day 40] alice:lisbon")

    # The second update sees the summary the first one produced.
    assert "summary text" in llm.calls[1]["user"]
    assert "[day 40] alice:lisbon" in llm.calls[1]["user"]


def test_ingest_strips_whitespace_from_the_updated_summary():
    baseline = SummaryBaseline(CannedLLM("  trimmed summary\n"))

    baseline.ingest("[day 0] alice:berlin")

    assert baseline.summary == "trimmed summary"


def test_answer_reads_only_from_the_running_summary():
    llm = CannedLLM("lisbon")
    baseline = SummaryBaseline(llm)
    baseline._summary = "alice now lives in lisbon"

    answer = baseline.answer("Where does alice live now?")

    assert answer == "lisbon"
    (call,) = llm.calls
    assert call["system"] == SUMMARY_ANSWER_SYSTEM_PROMPT
    assert "alice now lives in lisbon" in call["user"]


def test_answer_strips_whitespace_from_the_completion():
    baseline = SummaryBaseline(CannedLLM("  lisbon\n"))

    assert baseline.answer("Where now?") == "lisbon"


def test_answer_with_an_empty_summary_still_asks_the_model():
    llm = CannedLLM("unknown")
    baseline = SummaryBaseline(llm)

    answer = baseline.answer("Where does alice live now?")

    assert answer == "unknown"
    assert "(empty)" in llm.calls[0]["user"]


# --- the scored harness -------------------------------------------------------


def test_evaluate_summary_scores_every_gold_query():
    llm = RoutingFakeLLM()
    expected = sum(len(scenario.queries) for scenario in GOLD_SCENARIOS)

    report = evaluate_summary(llm)

    assert report.system == B2_SUMMARY
    assert len(report.outcomes) == expected


def test_evaluate_summary_rolls_up_accuracy_by_kind():
    # Grade every "changed" (evolution) question wrong, the rest right.
    llm = RoutingFakeLLM(correct_fn=lambda user: "changed" not in user.lower())

    by_kind = evaluate_summary(llm).accuracy_by_kind()

    assert by_kind[QueryKind.CURRENT] == 1.0
    assert by_kind[QueryKind.HISTORICAL] == 1.0
    assert by_kind[QueryKind.EVOLUTION] == 0.0


def test_evaluate_summary_overall_accuracy_matches_the_kinds():
    llm = RoutingFakeLLM(correct_fn=lambda user: "changed" not in user.lower())

    # current (3) + historical (2) right, evolution (3) wrong = 5 / 8.
    assert evaluate_summary(llm).accuracy == pytest.approx(5 / 8)


def test_evaluate_summary_records_answer_and_reference():
    llm = RoutingFakeLLM(answer="berlin")

    outcome = evaluate_summary(llm).outcomes[0]

    assert outcome.answer == "berlin"
    assert outcome.reference  # non-empty gold
    assert outcome.question


def test_evaluate_summary_accepts_a_separate_judge_client():
    answerer = RoutingFakeLLM(answer="x")
    judge_client = RoutingFakeLLM(correct_fn=lambda user: False)

    report = evaluate_summary(answerer, judge_client=judge_client)

    assert report.accuracy == 0.0
    # The answerer only updated and answered; the judge client only graded.
    assert all(c["system"] != JUDGE_SYSTEM_PROMPT for c in answerer.calls)
    assert all(c["system"] == JUDGE_SYSTEM_PROMPT for c in judge_client.calls)


def test_evaluate_summary_is_deterministic():
    first = evaluate_summary(RoutingFakeLLM())
    second = evaluate_summary(RoutingFakeLLM())

    assert [o.correct for o in first.outcomes] == [o.correct for o in second.outcomes]


def test_empty_report_has_zero_accuracy():
    report = evaluate_summary(RoutingFakeLLM(), scenarios=())

    assert report.accuracy == 0.0
    assert report.accuracy_by_kind() == {}


def test_format_baselines_lists_system_and_kinds():
    report = evaluate_summary(RoutingFakeLLM())

    rendered = format_baselines({B2_SUMMARY: report})

    assert B2_SUMMARY in rendered
    for kind in QueryKind:
        assert kind.value in rendered
