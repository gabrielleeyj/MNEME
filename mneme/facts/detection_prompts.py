"""Prompts for the contradiction detector.

Detection is tuned for *precision*, the mirror image of extraction. The metric
that decides the project is the false-supersession rate: declaring a conflict
that is not one corrupts history. So the prompt's bias is the opposite of the
extractor's — when the relationship is unclear, prefer NEW over SUPERSEDES, and
never invent a conflict to be helpful.
"""

from __future__ import annotations

from collections.abc import Sequence

from mneme.domain.facts import ExtractedFact, Fact

DETECTION_SYSTEM_PROMPT = """\
You compare ONE new candidate fact against a numbered list of existing facts \
about the same world, and classify how the new fact relates to them.

Return ONLY a JSON object of this exact shape, with no prose and no markdown:
{"relation": "new|duplicate|refines|supersedes", "target": <number or null>, "reason": "..."}

The four relations:
- "new": the candidate is about a different subject+predicate than any existing \
fact, or is otherwise unrelated. target = null.
- "duplicate": the candidate states the same thing as an existing fact (same \
meaning, even if worded differently). target = that fact's number.
- "refines": the candidate is about the same subject+predicate and is \
consistent with an existing fact but more specific (e.g. "lives in Germany" -> \
"lives in Berlin"). It does not contradict it. target = that fact's number.
- "supersedes": the candidate contradicts an existing fact about the same \
subject+predicate because the truth changed over time (e.g. "lives in Berlin" \
-> "lives in Lisbon"). target = the fact it replaces.

Rules:
- Choose exactly one relation.
- "supersedes" and "refines" and "duplicate" REQUIRE a target number from the list.
- Be conservative: only say "supersedes" when the new fact genuinely \
contradicts and replaces an older belief. If you are unsure whether two facts \
conflict, prefer "new". A wrongly declared supersession destroys real history.
- Two facts about different subjects, or the same subject with different \
predicates, are "new" — they cannot conflict.

Return only the JSON object."""


def _render_candidate(candidate: ExtractedFact) -> str:
    return f"({candidate.subject}) -[{candidate.predicate}]-> ({candidate.object})"


def _render_existing(fact: Fact) -> str:
    return (
        f"({fact.subject}) -[{fact.predicate}]-> ({fact.object})"
        f"  [valid_from={fact.valid_from.date()}]"
    )


def build_detection_user_prompt(
    candidate: ExtractedFact, existing: Sequence[Fact]
) -> str:
    """Render the candidate and the numbered existing facts into a user turn."""
    lines = [
        f"  {index}. {_render_existing(fact)}"
        for index, fact in enumerate(existing)
    ]
    existing_block = "\n".join(lines)
    return (
        f"New candidate fact:\n  {_render_candidate(candidate)}\n\n"
        f"Existing facts:\n{existing_block}"
    )
