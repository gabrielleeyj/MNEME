from __future__ import annotations

from mneme.embeddings.client import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingClient,
    FastEmbedEmbeddingClient,
)


def test_default_model():
    client = FastEmbedEmbeddingClient()
    assert client._model == DEFAULT_EMBEDDING_MODEL


def test_model_is_overridable():
    client = FastEmbedEmbeddingClient(model="BAAI/bge-base-en-v1.5")
    assert client._model == "BAAI/bge-base-en-v1.5"


def test_construction_does_not_load_the_model():
    # Building the client must be cheap: no weights downloaded or loaded until
    # the first embed call.
    client = FastEmbedEmbeddingClient()
    assert client._embedder is None


def test_empty_input_returns_empty_without_loading_model():
    client = FastEmbedEmbeddingClient()
    assert client.embed([]) == []
    assert client._embedder is None


def test_fastembed_client_satisfies_protocol():
    client = FastEmbedEmbeddingClient()
    assert isinstance(client, EmbeddingClient)
