"""Event domain model — the immutable units of the append-only log."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Actor(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class EventType(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    ARTIFACT = "artifact"
    REFLECTION = "reflection"


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable entry in the log. Never updated, never deleted."""

    event_id: int
    ts: datetime
    actor: Actor
    type: EventType
    content: str
    parent_event_id: int | None = None
