"""Embedding client seam.

``EmbeddingClient`` is the narrow contract the semantic index depends on, kept
to a single batch ``embed`` so tests can inject a deterministic fake.
``OpenAIEmbeddingClient`` is the concrete implementation; ``openai`` is a lazy
import so the package installs and the suite runs without it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# text-embedding-3-small: 1536 dims, cheap, strong enough to separate the
# contradiction/evolution pairs the thesis turns on. Override for a stronger
# model if recall on the synthetic set demands it.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text, in the same order."""
        ...


class OpenAIEmbeddingClient:
    """Thin wrapper over the OpenAI embeddings API.

    The API key is validated at construction so a misconfiguration fails fast,
    before any indexing work begins.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not set; pass api_key= or set the env var"
            )
        self._model = model
        self._api_key = resolved_key
        self._client = None  # lazily constructed on first use

    def _ensure_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "the 'openai' package is required for OpenAIEmbeddingClient; "
                    "install with: pip install 'mneme[embeddings]'"
                ) from exc
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        response = client.embeddings.create(model=self._model, input=list(texts))
        return [list(item.embedding) for item in response.data]
