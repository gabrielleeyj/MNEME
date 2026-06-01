"""Coax a JSON object out of an LLM completion.

Both the fact extractor and the contradiction detector ask the model for a
single JSON object and get back text that may be fenced in ```json blocks or
wrapped in prose. This shared parser strips the fences and, failing a clean
parse, falls back to the outermost ``{...}`` span. It raises
``JSONExtractionError`` so each caller can re-wrap it in its own domain error.
"""

from __future__ import annotations

import json
from typing import Any


class JSONExtractionError(ValueError):
    """No well-formed JSON object could be recovered from the text."""


def extract_json_object(raw: str) -> Any:
    """Parse the JSON object in ``raw``, tolerating code fences and prose."""
    text = _strip_code_fences(raw.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Tolerate prose around the object: take the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JSONExtractionError(f"no JSON object found in response: {raw!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JSONExtractionError(
            f"could not parse JSON from response: {raw!r}"
        ) from exc


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
