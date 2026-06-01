"""Embedding client seam.

``EmbeddingClient`` is the narrow contract the semantic index depends on, kept
to a single batch ``embed`` so tests can inject a deterministic fake.

``FastEmbedEmbeddingClient`` is the concrete implementation: a local ONNX model
(no API key, no network after the one-time model download, no torch). Anthropic
has no embeddings endpoint, so embeddings come from here rather than the same
provider as the LLM. ``fastembed`` is a lazy import — and the model is loaded
lazily on first use, since loading weights is expensive — so the package
installs and the suite runs without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# bge-small-en-v1.5: 384 dims, strong retrieval quality for its size, and
# fastembed's well-trodden default. Override for a larger model if recall on the
# synthetic set demands it.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text, in the same order."""
        ...


class FastEmbedEmbeddingClient:
    """Local embeddings via fastembed (ONNX). No API key required.

    The model is downloaded (once) and loaded on first ``embed`` call, not at
    construction, so building the client is cheap and side-effect free.
    """

    def __init__(self, *, model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model = model
        self._embedder = None  # lazily constructed on first use

    def _ensure_embedder(self):
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "the 'fastembed' package is required for "
                    "FastEmbedEmbeddingClient; install with: "
                    "pip install 'mneme[embeddings]'"
                ) from exc
            self._embedder = TextEmbedding(model_name=self._model)
        return self._embedder

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embedder = self._ensure_embedder()
        return [[float(x) for x in vector] for vector in embedder.embed(list(texts))]
