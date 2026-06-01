"""Embedding seam: text -> dense vector, behind a narrow protocol.

Workstream 4. The protocol lets the semantic index take a test double with
hand-built vectors, so the index's retrieval behaviour can be exercised without
a network or an API key.
"""

from mneme.embeddings.client import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingClient,
    FastEmbedEmbeddingClient,
)

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingClient",
    "FastEmbedEmbeddingClient",
]
