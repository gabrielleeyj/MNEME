from __future__ import annotations

import pytest

from mneme.llm.client import DEFAULT_MODEL, AnthropicClient, LLMClient


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_explicit_api_key_is_accepted_without_network(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key="sk-test")
    # Constructing must not build the SDK client or touch the network.
    assert client._client is None
    assert client._model == DEFAULT_MODEL


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    client = AnthropicClient()
    assert client._api_key == "sk-from-env"


def test_model_is_overridable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key="sk-test", model="claude-sonnet-4-6")
    assert client._model == "claude-sonnet-4-6"


def test_anthropic_client_satisfies_protocol(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient(api_key="sk-test")
    assert isinstance(client, LLMClient)
