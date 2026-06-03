"""Grade a baseline's free-text answer against the gold reference.

MNEME's own outputs are scored by exact match on structured objects, but a RAG
baseline answers in prose, so its answers need a grader that judges meaning over
wording. That is an LLM-as-judge — the standard for free-text memory evals — and
it is the live-usage scoring path the B1 numbers are reported under.

The verdict is untrusted model output: it is parsed strictly into a boolean and
raises ``JudgeError`` rather than guessing, so a malformed grade never silently
counts as correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mneme.baselines.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from mneme.llm.client import LLMClient
from mneme.llm.json_io import JSONExtractionError, extract_json_object

__all__ = ["JudgeError", "Verdict", "LLMJudge"]


class JudgeError(ValueError):
    """The judge's response could not be parsed into a boolean verdict."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether an answer was judged correct, with the judge's stated reason."""

    correct: bool
    reason: str


class LLMJudge:
    """Score a candidate answer against a gold reference with one LLM call."""

    def __init__(self, client: LLMClient, *, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def judge(
        self, question: str, reference: Sequence[str], answer: str
    ) -> Verdict:
        raw = self._client.complete(
            system=JUDGE_SYSTEM_PROMPT,
            user=build_judge_user_prompt(question, reference, answer),
            max_tokens=self._max_tokens,
        )
        return _parse_verdict(raw)


def _parse_verdict(raw: str) -> Verdict:
    try:
        data = extract_json_object(raw)
    except JSONExtractionError as exc:
        raise JudgeError(str(exc)) from exc
    if not isinstance(data, dict):
        raise JudgeError(f"judge response is not a JSON object: {raw!r}")

    correct = _parse_correct(data.get("correct"), raw)
    reason = data.get("reason")
    reason_text = reason.strip() if isinstance(reason, str) else ""
    return Verdict(correct, reason_text)


def _parse_correct(raw_correct: Any, raw: str) -> bool:
    if not isinstance(raw_correct, bool):
        raise JudgeError(
            f"judge response missing boolean 'correct': {raw!r}"
        )
    return raw_correct
