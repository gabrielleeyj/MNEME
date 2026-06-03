"""B2 — running summary: compress the timeline into prose, answer from that.

The other naive baseline. Where B1 keeps every raw message and retrieves, B2
keeps *nothing* raw: each message is folded into a single running natural-
language summary, and questions are answered from that summary alone. This is
the summarization-memory straw man, and its defining property is lossy
compression — a concise summary tends to drop superseded detail and exact dates,
so a question about the past or about change over time is only answerable if the
summary happened to retain it. That is precisely the failure MNEME's structured
supersession avoids.

The summary update and the answer are the only LLM calls; the client is
injected, so tests run offline with a scripted fake.
"""

from __future__ import annotations

from mneme.baselines.prompts import (
    SUMMARY_ANSWER_SYSTEM_PROMPT,
    SUMMARY_UPDATE_SYSTEM_PROMPT,
    build_summary_answer_user_prompt,
    build_summary_update_user_prompt,
)
from mneme.llm.client import LLMClient

__all__ = ["SummaryBaseline"]


class SummaryBaseline:
    """Fold each message into one running summary; answer from the summary."""

    def __init__(self, client: LLMClient, *, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._summary = ""

    @property
    def summary(self) -> str:
        """The running summary as it stands."""
        return self._summary

    def ingest(self, message: str) -> None:
        """Fold one message into the running summary (one LLM call)."""
        self._summary = self._client.complete(
            system=SUMMARY_UPDATE_SYSTEM_PROMPT,
            user=build_summary_update_user_prompt(self._summary, message),
            max_tokens=self._max_tokens,
        ).strip()

    def answer(self, question: str) -> str:
        """Return the model's free-text answer from the running summary alone."""
        raw = self._client.complete(
            system=SUMMARY_ANSWER_SYSTEM_PROMPT,
            user=build_summary_answer_user_prompt(self._summary, question),
            max_tokens=self._max_tokens,
        )
        return raw.strip()
