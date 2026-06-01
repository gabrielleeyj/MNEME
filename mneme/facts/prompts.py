"""Prompts for the fact extractor.

Extraction is tuned for *recall*: pull every durable fact, decompose compound
statements, and let the downstream contradiction detector (workstream 3) and its
precision-tuned judgment decide what conflicts. Missing a fact here is a loss the
rest of the pipeline cannot recover; an over-eager extraction can still be
filtered later.
"""

from __future__ import annotations

from mneme.domain.events import Event

EXTRACTION_SYSTEM_PROMPT = """\
You extract atomic, durable facts from a single conversational message.

Return ONLY a JSON object of this exact shape, with no prose and no markdown:
{"facts": [{"subject": "...", "predicate": "...", "object": "...", "valid_from": "...", "confidence": 0.0}]}

Rules:
- subject / object: short noun phrases. Use the speaker's stable identity (e.g. "user") as the subject when the message is about themselves.
- predicate: a short relation in snake_case (e.g. lives_in, works_at, prefers, owns, named).
- valid_from: an ISO 8601 date or datetime for when the fact became true, ONLY if the message states or clearly implies it. Otherwise use null.
- confidence: your confidence in the fact, from 0.0 to 1.0.
- Decompose compound statements into multiple separate facts.
- Favor recall: extract every stable, durable fact about the speaker or the world. Skip pure pleasantries, questions, and ephemeral chit-chat that assert nothing lasting.
- If the message asserts no durable facts, return {"facts": []}.

Return only the JSON object."""


def build_extraction_user_prompt(event: Event) -> str:
    """Render one event into the user turn for extraction."""
    return (
        f"Message timestamp: {event.ts.isoformat()}\n"
        f"Speaker: {event.actor.value}\n"
        f"Message:\n{event.content}"
    )
