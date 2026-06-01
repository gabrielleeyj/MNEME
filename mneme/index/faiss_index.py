"""A FAISS HNSW index that speaks cosine and external ids.

Two design choices, coupled on purpose:

  * **Metric.** HNSW with ``METRIC_INNER_PRODUCT`` over L2-normalized vectors
    gives cosine similarity. Normalization is done here, on add and on query,
    so callers get cosine without having to remember to normalize.
  * **Ids.** FAISS returns its own sequential row numbers. We keep a parallel
    list mapping those back to the caller's external ids (fact ids, event ids),
    so the rest of the system never sees a FAISS-internal index.

``faiss`` and ``numpy`` are imported lazily so the package installs and the
core suite runs without the vector extra.
"""

from __future__ import annotations

from collections.abc import Sequence

# HNSW graph degree. 32 is the common default: enough connectivity for good
# recall on the small synthetic set without blowing up build time.
DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64


class FaissHnswIndex:
    def __init__(
        self,
        dim: int,
        *,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
    ) -> None:
        if dim <= 0:
            raise ValueError("embedding dimension must be positive")
        try:
            import faiss
            import numpy as np
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "the 'faiss-cpu' and 'numpy' packages are required for "
                "FaissHnswIndex; install with: pip install 'mneme[vectors]'"
            ) from exc
        self._faiss = faiss
        self._np = np
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        self._index = index
        self._dim = dim
        self._ids: list[int] = []

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, external_id: int, vector: Sequence[float]) -> None:
        self.add_many([external_id], [vector])

    def add_many(
        self,
        external_ids: Sequence[int],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(external_ids) != len(vectors):
            raise ValueError("external_ids and vectors must be the same length")
        if not vectors:
            return
        prepared = self._prepare(vectors)
        self._index.add(prepared)
        self._ids.extend(external_ids)

    def search(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(external_id, cosine_score)`` pairs, best first."""
        if k <= 0:
            raise ValueError("k must be positive")
        if not self._ids:
            return []
        prepared = self._prepare([vector])
        scores, indices = self._index.search(prepared, min(k, len(self._ids)))
        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS pads with -1 when fewer than k neighbours exist
                continue
            results.append((self._ids[int(idx)], float(score)))
        return results

    def _prepare(self, vectors: Sequence[Sequence[float]]):
        arr = self._np.asarray(vectors, dtype=self._np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._dim:
            raise ValueError(
                f"expected vectors of dimension {self._dim}, got shape {arr.shape}"
            )
        # Copy to a contiguous buffer; normalize_L2 mutates in place.
        arr = self._np.ascontiguousarray(arr)
        self._faiss.normalize_L2(arr)
        return arr

    def __len__(self) -> int:
        return len(self._ids)
