"""LLM-backed fact extractor (workstream 2).

Turns one event into ``(subject, predicate, object, valid_from)`` candidates by
prompting an ``LLMClient`` and parsing its JSON. Implements the ``Extractor``
protocol, so it drops straight into ``FactStore.rebuild`` and every baseline —
one extractor shared everywhere keeps the comparison honest.

The response is untrusted external data, so parsing validates strictly and
raises ``ExtractionError`` with the offending payload rather than silently
dropping or guessing. That visibility is what makes prompt-tuning tractable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from mneme.domain.events import Event
from mneme.domain.facts import ExtractedFact
from mneme.facts.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from mneme.llm.client import LLMClient

_REQUIRED_FIELDS = ("subject", "predicate", "object")


class ExtractionError(ValueError):
    """The LLM response could not be parsed into well-formed facts."""


class LLMExtractor:
    def __init__(self, client: LLMClient, *, max_tokens: int | None = None) -> None:
        self._client = client
        self._max_tokens = max_tokens

    def extract(self, event: Event) -> Sequence[ExtractedFact]:
        raw = self._client.complete(
            system=EXTRACTION_SYSTEM_PROMPT,
            user=build_extraction_user_prompt(event),
            max_tokens=self._max_tokens,
        )
        entries = _parse_facts_payload(raw)
        return tuple(self._to_fact(entry, event) for entry in entries)

    def _to_fact(self, entry: dict[str, Any], event: Event) -> ExtractedFact:
        if not isinstance(entry, dict):
            raise ExtractionError(f"fact entry is not an object: {entry!r}")
        values = {}
        for field in _REQUIRED_FIELDS:
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ExtractionError(
                    f"fact entry missing non-empty '{field}': {entry!r}"
                )
            values[field] = value.strip()

        return ExtractedFact(
            subject=values["subject"],
            predicate=values["predicate"],
            object=values["object"],
            valid_from=_parse_valid_from(entry.get("valid_from"), default=event.ts),
            confidence=_parse_confidence(entry.get("confidence")),
        )


def _parse_facts_payload(raw: str) -> list[Any]:
    data = _load_json_object(raw)
    if not isinstance(data, dict) or "facts" not in data:
        raise ExtractionError(f"response missing 'facts' key: {raw!r}")
    facts = data["facts"]
    if not isinstance(facts, list):
        raise ExtractionError(f"'facts' is not a list: {facts!r}")
    return facts


def _load_json_object(raw: str) -> Any:
    text = _strip_code_fences(raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Tolerate prose around the object: take the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ExtractionError(f"no JSON object found in response: {raw!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"could not parse JSON from response: {raw!r}") from exc


def _strip_code_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    body = text[3:]
    if body.startswith("json"):
        body = body[4:]
    closing = body.rfind("```")
    if closing != -1:
        body = body[:closing]
    return body.strip()


def _parse_valid_from(raw: Any, *, default: datetime) -> datetime:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    if not isinstance(raw, str):
        raise ExtractionError(f"valid_from is not a string or null: {raw!r}")
    try:
        parsed = datetime.fromisoformat(raw.strip())
    except ValueError as exc:
        raise ExtractionError(f"valid_from is not ISO 8601: {raw!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_confidence(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ExtractionError(f"confidence is not a number or null: {raw!r}")
    return float(raw)
