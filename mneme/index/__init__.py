"""Semantic index: nearest-neighbour retrieval over embedded text.

Workstream 4. This is what gives workstream 3's contradiction detector a real
top-k of semantically related candidates to judge, instead of a brute-force
scan of every stored fact.
"""

from mneme.index.faiss_index import FaissHnswIndex
from mneme.index.render import embedding_text_for_event, embedding_text_for_fact
from mneme.index.semantic_index import SemanticIndex

__all__ = [
    "FaissHnswIndex",
    "SemanticIndex",
    "embedding_text_for_event",
    "embedding_text_for_fact",
]
