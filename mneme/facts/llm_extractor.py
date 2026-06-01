"""LLM-backed fact extractor (workstream 2).

Turns one event into ``(subject, predicate, object, valid_from)`` candidates by
prompting an ``LLMClient`` and parsing its JSON. Implements the ``Extractor``
protocol, so it drops straight into ``FactStore.rebuild`` and every baseline —
one extractor shared everywhere keeps the comparison honest.

The response is untrusted external data. The *required* triple
(subject/predicate/object) is validated strictly: a malformed or empty value
raises ``ExtractionError`` with the offending payload rather than guessing,
because a fact without them is useless. The *optional* ``valid_from`` is
best-effort: it already defaults to the event timestamp when absent, so an
unrecognized date degrades to that default with an ``ExtractionWarning`` instead
of crashing — one fuzzy date must not abort a whole extraction pass.
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from mneme.domain.events import Event
from mneme.domain.facts import ExtractedFact
from mneme.facts.prompts import EXTRACTION_SYSTEM_PROMPT, build_extraction_user_prompt
from mneme.llm.client import LLMClient

_REQUIRED_FIELDS = ("subject", "predicate", "object")

# Coarse temporal forms the model emits that are not ISO 8601 calendar dates.
# Each resolves to the first instant of the period (the earliest the fact could
# have become valid).
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")
_QUARTER_RE = re.compile(r"^(\d{4})-?Q([1-4])$", re.IGNORECASE)


class ExtractionError(ValueError):
    """The LLM response could not be parsed into well-formed facts."""


class ExtractionWarning(UserWarning):
    """A non-fatal anomaly in an extraction (e.g. an unrecognized valid_from)."""


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
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default

    parsed = _try_iso_datetime(text) or _try_coarse_date(text)
    if parsed is None:
        warnings.warn(
            f"valid_from {raw!r} is not a recognized date; "
            "defaulting to the event timestamp",
            ExtractionWarning,
            stacklevel=2,
        )
        return default
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _try_iso_datetime(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _try_coarse_date(text: str) -> datetime | None:
    """Resolve year / year-month / year-quarter to the first day of the period."""
    try:
        match = _QUARTER_RE.match(text)
        if match:
            month = (int(match.group(2)) - 1) * 3 + 1
            return datetime(int(match.group(1)), month, 1, tzinfo=timezone.utc)
        match = _YEAR_MONTH_RE.match(text)
        if match:
            return datetime(
                int(match.group(1)), int(match.group(2)), 1, tzinfo=timezone.utc
            )
        match = _YEAR_RE.match(text)
        if match:
            return datetime(int(match.group(1)), 1, 1, tzinfo=timezone.utc)
    except ValueError:
        return None
    return None


def _parse_confidence(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ExtractionError(f"confidence is not a number or null: {raw!r}")
    return float(raw)
