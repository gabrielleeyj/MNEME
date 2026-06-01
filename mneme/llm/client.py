"""LLM client seam.

One client serves both jobs in the plan: fact extraction (workstream 2) and
contradiction judgment (workstream 3). They are the *same model* run at
different operating points — extraction tuned for recall, judgment for
precision — so the difference lives in the prompts and thresholds the callers
pass, not here.

``LLMClient`` is the contract callers depend on, kept narrow so tests can inject
a fake. ``AnthropicClient`` is the concrete implementation; ``anthropic`` is a
lazy import so the package installs and the test suite runs without it.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

# Haiku 4.5: cheap enough to run over the whole synthetic set in one extraction
# pass, capable enough for structured extraction. Override per use case (e.g. a
# stronger model for the precision-critical contradiction judge).
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return the model's text completion for a system + user prompt."""
        ...


class AnthropicClient:
    """Thin wrapper over the Anthropic Messages API.

    The system prompt is marked for prompt caching: it is identical across every
    extraction/judgment call, so caching it avoids re-billing those tokens on
    each request.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set; pass api_key= or set the env var"
            )
        self._model = model
        self._api_key = resolved_key
        self._max_tokens = max_tokens
        self._client = None  # lazily constructed on first use

    def _ensure_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "the 'anthropic' package is required for AnthropicClient; "
                    "install with: pip install 'mneme[llm]'"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(self, *, system: str, user: str, max_tokens: int | None = None) -> str:
        client = self._ensure_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=max_tokens or self._max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        )
