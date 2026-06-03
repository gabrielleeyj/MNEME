"""B1 raw RAG: retrieval + answer, the LLM judge, and the scored rollups.

All offline: a bag-of-words fake embedder (``fake_embedder`` fixture) gives real
retrieval behaviour with no network, and a routing fake LLM stands in for both
the answerer and the judge so the harness is exercised deterministically.
"""

from __future__ import annotations

import json

import pytest

from mneme.baselines.harness import B1_RAW_RAG, evaluate_rag, format_baselines
from mneme.baselines.judge import JudgeError, LLMJudge, Verdict
from mneme.baselines.prompts import JUDGE_SYSTEM_PROMPT, RAG_ANSWER_SYSTEM_PROMPT
from mneme.baselines.rag import RawRagBaseline
from mneme.eval.dataset import GOLD_SCENARIOS
from mneme.eval.scenario import QueryKind
from mneme.index.semantic_index import SemanticIndex


class CannedLLM:
    """Returns one fixed completion and records every call. No network."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


class RoutingFakeLLM:
    """Answers on the RAG prompt, grades on the judge prompt — deterministically.

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
        return self.answer


# --- retrieval + answer (B1) --------------------------------------------------


def test_answer_feeds_retrieved_dated_snippets_to_the_model(fake_embedder):
    llm = CannedLLM("lisbon")
    rag = RawRagBaseline(SemanticIndex(fake_embedder), llm)
    rag.ingest(1, "alice berlin", display="[day 0] alice:berlin")
    rag.ingest(2, "alice lisbon", display="[day 40] alice:lisbon")

    answer = rag.answer("where does alice live")

    assert answer == "lisbon"
    (call,) = llm.calls
    assert call["system"] == RAG_ANSWER_SYSTEM_PROMPT
    # Both messages retrieved and shown in chronological (insertion) order.
    assert "[day 0] alice:berlin" in call["user"]
    assert "[day 40] alice:lisbon" in call["user"]
    assert call["user"].index("[day 0]") < call["user"].index("[day 40]")


def test_answer_strips_whitespace_from_the_completion(fake_embedder):
    rag = RawRagBaseline(SemanticIndex(fake_embedder), CannedLLM("  berlin\n"))
    rag.ingest(1, "alice berlin", display="[day 0] alice:berlin")

    assert rag.answer("where does alice live") == "berlin"


def test_answer_with_no_messages_indexed_still_asks_the_model(fake_embedder):
    llm = CannedLLM("unknown")
    rag = RawRagBaseline(SemanticIndex(fake_embedder), llm)

    answer = rag.answer("where does alice live")

    assert answer == "unknown"
    assert "(no messages retrieved)" in llm.calls[0]["user"]


# --- the LLM judge ------------------------------------------------------------


def test_judge_parses_a_true_verdict():
    judge = LLMJudge(CannedLLM('{"correct": true, "reason": "match"}'))

    verdict = judge.judge("Where now?", ("lisbon",), "Lisbon")

    assert verdict == Verdict(True, "match")


def test_judge_parses_a_false_verdict():
    judge = LLMJudge(CannedLLM('{"correct": false, "reason": "stale"}'))

    assert judge.judge("Where now?", ("lisbon",), "Berlin").correct is False


def test_judge_tolerates_fenced_json():
    judge = LLMJudge(CannedLLM('```json\n{"correct": true, "reason": "ok"}\n```'))

    assert judge.judge("Q", ("a",), "a").correct is True


def test_judge_rejects_missing_correct_field():
    judge = LLMJudge(CannedLLM('{"reason": "no verdict"}'))

    with pytest.raises(JudgeError):
        judge.judge("Q", ("a",), "a")


def test_judge_rejects_non_boolean_correct():
    judge = LLMJudge(CannedLLM('{"correct": "yes"}'))

    with pytest.raises(JudgeError):
        judge.judge("Q", ("a",), "a")


def test_judge_rejects_a_non_object_response():
    judge = LLMJudge(CannedLLM("[1, 2, 3]"))

    with pytest.raises(JudgeError):
        judge.judge("Q", ("a",), "a")


def test_judge_rejects_unparseable_text():
    judge = LLMJudge(CannedLLM("I think it is correct"))

    with pytest.raises(JudgeError):
        judge.judge("Q", ("a",), "a")


# --- the scored harness -------------------------------------------------------


def test_evaluate_rag_scores_every_gold_query():
    llm = RoutingFakeLLM()
    expected = sum(len(scenario.queries) for scenario in GOLD_SCENARIOS)

    report = evaluate_rag(_embedder(), llm)

    assert report.system == B1_RAW_RAG
    assert len(report.outcomes) == expected


def test_evaluate_rag_rolls_up_accuracy_by_kind():
    # Grade every "changed" (evolution) question wrong, the rest right.
    llm = RoutingFakeLLM(correct_fn=lambda user: "changed" not in user.lower())

    by_kind = evaluate_rag(_embedder(), llm).accuracy_by_kind()

    assert by_kind[QueryKind.CURRENT] == 1.0
    assert by_kind[QueryKind.HISTORICAL] == 1.0
    assert by_kind[QueryKind.EVOLUTION] == 0.0


def test_evaluate_rag_overall_accuracy_matches_the_kinds():
    llm = RoutingFakeLLM(correct_fn=lambda user: "changed" not in user.lower())

    # current (3) + historical (2) right, evolution (3) wrong = 5 / 8.
    assert evaluate_rag(_embedder(), llm).accuracy == pytest.approx(5 / 8)


def test_evaluate_rag_records_answer_and_reference():
    llm = RoutingFakeLLM(answer="berlin")

    outcome = evaluate_rag(_embedder(), llm).outcomes[0]

    assert outcome.answer == "berlin"
    assert outcome.reference  # non-empty gold
    assert outcome.question


def test_evaluate_rag_accepts_a_separate_judge_client():
    answerer = RoutingFakeLLM(answer="x")
    judge_client = RoutingFakeLLM(correct_fn=lambda user: False)

    report = evaluate_rag(_embedder(), answerer, judge_client=judge_client)

    assert report.accuracy == 0.0
    # The answerer never graded; the judge client never answered.
    assert all(c["system"] == RAG_ANSWER_SYSTEM_PROMPT for c in answerer.calls)
    assert all(c["system"] == JUDGE_SYSTEM_PROMPT for c in judge_client.calls)


def test_evaluate_rag_is_deterministic():
    first = evaluate_rag(_embedder(), RoutingFakeLLM())
    second = evaluate_rag(_embedder(), RoutingFakeLLM())

    assert [o.correct for o in first.outcomes] == [o.correct for o in second.outcomes]


def test_empty_report_has_zero_accuracy():
    report = evaluate_rag(_embedder(), RoutingFakeLLM(), scenarios=())

    assert report.accuracy == 0.0
    assert report.accuracy_by_kind() == {}


def test_format_baselines_lists_system_and_kinds():
    report = evaluate_rag(_embedder(), RoutingFakeLLM())

    rendered = format_baselines({B1_RAW_RAG: report})

    assert B1_RAW_RAG in rendered
    for kind in QueryKind:
        assert kind.value in rendered


def _embedder():
    from tests.conftest import FakeEmbeddingClient

    return FakeEmbeddingClient()
