"""B1 — raw RAG: retrieve the nearest messages, let the LLM answer from them.

The naive baseline the thesis is measured against. It keeps no facts and no
history: every message is embedded as-is, a question retrieves the top-k nearest
messages, and the model answers from that text alone. There is no supersession
and no valid-time index, so a question about the past or about change over time
is only answerable if the raw messages happen to be retrieved and the model
reasons the timeline out unaided — which is exactly the weakness MNEME claims to
fix.

Retrieval (embeddings + FAISS) reuses the workstream-4 ``SemanticIndex``; the
answer step is the only LLM call. Both seams are injected, so tests run offline
with a fake embedder and a scripted client.
"""

from __future__ import annotations

from mneme.baselines.prompts import (
    RAG_ANSWER_SYSTEM_PROMPT,
    build_rag_answer_user_prompt,
)
from mneme.index.semantic_index import SemanticIndex
from mneme.llm.client import LLMClient

__all__ = ["DEFAULT_TOP_K", "RawRagBaseline"]

# The synthetic scenarios are short (a handful of messages each), so a generous
# k retrieves essentially the whole timeline: B1's failure is meant to be
# reasoning over retrieved text, not recall, so retrieval is not the bottleneck.
DEFAULT_TOP_K = 10


class RawRagBaseline:
    """Embed messages, retrieve the nearest, answer the question from them."""

    def __init__(
        self,
        index: SemanticIndex,
        client: LLMClient,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int | None = None,
    ) -> None:
        self._index = index
        self._client = client
        self._top_k = top_k
        self._max_tokens = max_tokens
        self._display: dict[int, str] = {}

    def ingest(self, doc_id: int, text: str, *, display: str | None = None) -> None:
        """Index one message. ``text`` is embedded; ``display`` (default ``text``)
        is what the model is shown at answer time."""
        self._index.add(doc_id, text)
        self._display[doc_id] = display if display is not None else text

    def answer(self, question: str) -> str:
        """Retrieve the nearest messages and return the model's free-text answer."""
        hits = self._index.search(question, k=self._top_k)
        # Present the retrieved messages as a timeline (by insertion order =
        # chronological), not by similarity score, so the model sees the history
        # in the order it happened.
        doc_ids = sorted(doc_id for doc_id, _score in hits)
        snippets = [self._display[doc_id] for doc_id in doc_ids]
        raw = self._client.complete(
            system=RAG_ANSWER_SYSTEM_PROMPT,
            user=build_rag_answer_user_prompt(question, snippets),
            max_tokens=self._max_tokens,
        )
        return raw.strip()
