from __future__ import annotations

import pytest

from mneme.embeddings.client import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingClient,
    OpenAIEmbeddingClient,
)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIEmbeddingClient()


def test_explicit_api_key_is_accepted_without_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIEmbeddingClient(api_key="sk-test")
    # Constructing must not build the SDK client or touch the network.
    assert client._client is None
    assert client._model == DEFAULT_EMBEDDING_MODEL


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    client = OpenAIEmbeddingClient()
    assert client._api_key == "sk-from-env"


def test_model_is_overridable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIEmbeddingClient(api_key="sk-test", model="text-embedding-3-large")
    assert client._model == "text-embedding-3-large"


def test_empty_input_returns_empty_without_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIEmbeddingClient(api_key="sk-test")
    # No texts means no work — and crucially no SDK construction.
    assert client.embed([]) == []
    assert client._client is None


def test_openai_client_satisfies_protocol(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIEmbeddingClient(api_key="sk-test")
    assert isinstance(client, EmbeddingClient)
