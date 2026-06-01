from __future__ import annotations

import math

import pytest

pytest.importorskip("faiss")
pytest.importorskip("numpy")

from mneme.index.faiss_index import FaissHnswIndex


def test_empty_index_search_returns_empty():
    index = FaissHnswIndex(dim=3)
    assert index.search([1.0, 0.0, 0.0], k=5) == []
    assert len(index) == 0


def test_returns_nearest_by_cosine():
    index = FaissHnswIndex(dim=3)
    index.add(10, [1.0, 0.0, 0.0])
    index.add(20, [0.0, 1.0, 0.0])
    index.add(30, [0.0, 0.0, 1.0])

    hits = index.search([0.9, 0.1, 0.0], k=3)

    assert len(hits) == 3
    assert hits[0][0] == 10  # closest to the x-axis vector
    # Scores are cosine: top hit is near 1, last is near 0.
    assert hits[0][1] == pytest.approx(0.9 / math.hypot(0.9, 0.1), rel=1e-3)


def test_normalization_makes_magnitude_irrelevant():
    index = FaissHnswIndex(dim=2)
    index.add(1, [1.0, 0.0])
    index.add(2, [0.0, 5.0])  # long vector, but orthogonal to the query

    # A long query along x still matches id 1 (cosine ignores magnitude).
    hits = index.search([100.0, 0.0], k=1)
    assert hits[0][0] == 1
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_k_is_clamped_to_index_size():
    index = FaissHnswIndex(dim=2)
    index.add(1, [1.0, 0.0])
    index.add(2, [0.0, 1.0])

    hits = index.search([1.0, 1.0], k=10)
    assert len(hits) == 2


def test_add_many_matches_individual_adds():
    index = FaissHnswIndex(dim=2)
    index.add_many([1, 2], [[1.0, 0.0], [0.0, 1.0]])
    assert len(index) == 2
    assert index.search([1.0, 0.0], k=1)[0][0] == 1


def test_external_ids_are_preserved_not_faiss_row_numbers():
    index = FaissHnswIndex(dim=2)
    index.add(999, [1.0, 0.0])  # external id far from FAISS row 0
    assert index.search([1.0, 0.0], k=1)[0][0] == 999


def test_dimension_mismatch_raises():
    index = FaissHnswIndex(dim=3)
    with pytest.raises(ValueError, match="dimension"):
        index.add(1, [1.0, 0.0])


def test_add_many_length_mismatch_raises():
    index = FaissHnswIndex(dim=2)
    with pytest.raises(ValueError, match="same length"):
        index.add_many([1, 2], [[1.0, 0.0]])


def test_non_positive_dim_raises():
    with pytest.raises(ValueError, match="dimension"):
        FaissHnswIndex(dim=0)


def test_non_positive_k_raises():
    index = FaissHnswIndex(dim=2)
    index.add(1, [1.0, 0.0])
    with pytest.raises(ValueError, match="k must be positive"):
        index.search([1.0, 0.0], k=0)
