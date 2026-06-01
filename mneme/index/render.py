"""What text we hand the embedder for each domain object.

This choice is load-bearing for the thesis: the contradiction detector only
sees facts that this rendering pulls together in vector space. It lives in one
place, as one-liners, so workstream 3 can tune the granularity (whole triple
vs. subject+predicate) without touching the index or the embedding client.
"""

from __future__ import annotations

from typing import Protocol


class _Triple(Protocol):
    subject: str
    predicate: str
    object: str


class _HasContent(Protocol):
    content: str


def embedding_text_for_fact(fact: _Triple) -> str:
    """Render a fact (extracted or stored) as the text we embed.

    Default: the full triple. Two facts contradict only if they are about the
    same subject+predicate, so the subject and predicate must be in the text;
    the object carries the value that may have changed.
    """
    return f"{fact.subject} {fact.predicate} {fact.object}"


def embedding_text_for_event(event: _HasContent) -> str:
    """Render an event as the text we embed — its raw content."""
    return event.content
