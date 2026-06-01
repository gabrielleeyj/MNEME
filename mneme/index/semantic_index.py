"""Embed text, index it, search it — the workstream-4 deliverable.

``SemanticIndex`` wires an ``EmbeddingClient`` to a ``FaissHnswIndex``. It is
deliberately domain-agnostic: it stores ``(external_id, text)`` pairs and
returns ``(external_id, score)`` hits. Callers turn facts or events into text
with the helpers in ``mneme.index.render``, so this class never imports the
domain models and workstream 3 can re-tune what gets embedded in one place.

The FAISS index is built lazily on the first add, because the embedding
dimension is not known until the client returns its first vector.
"""

from __future__ import annotations

from collections.abc import Sequence

from mneme.embeddings.client import EmbeddingClient
from mneme.index.faiss_index import DEFAULT_M, FaissHnswIndex


class SemanticIndex:
    def __init__(self, embedding_client: EmbeddingClient, *, m: int = DEFAULT_M) -> None:
        self._embed = embedding_client
        self._m = m
        self._index: FaissHnswIndex | None = None

    def add(self, external_id: int, text: str) -> None:
        self.add_many([(external_id, text)])

    def add_many(self, items: Sequence[tuple[int, str]]) -> None:
        if not items:
            return
        ids = [external_id for external_id, _ in items]
        texts = [text for _, text in items]
        vectors = self._embed.embed(texts)
        if len(vectors) != len(texts):
            raise ValueError(
                "embedding client returned the wrong number of vectors: "
                f"expected {len(texts)}, got {len(vectors)}"
            )
        self._ensure_index(len(vectors[0]))
        assert self._index is not None
        self._index.add_many(ids, vectors)

    def search(self, text: str, k: int = 10) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(external_id, score)`` hits for ``text``."""
        if self._index is None:
            return []
        vectors = self._embed.embed([text])
        if not vectors:
            return []
        return self._index.search(vectors[0], k)

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self._index = FaissHnswIndex(dim, m=self._m)

    def __len__(self) -> int:
        return 0 if self._index is None else len(self._index)
