"""Prompts for the B1 raw-RAG baseline: one to answer, one to judge.

Both are deliberately plain. B1 is the naive-RAG straw man the thesis must
beat, so its answerer gets only the retrieved messages and no notion of
supersession or valid-time — exactly the system MNEME claims to improve on. The
judge is shared scoring machinery, kept separate so the same grader could score
any baseline's free-text answer.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "RAG_ANSWER_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "build_rag_answer_user_prompt",
    "build_judge_user_prompt",
]


RAG_ANSWER_SYSTEM_PROMPT = (
    "You answer questions about a person using only the chat messages provided. "
    "Each message is tagged with the day it was sent, like '[day 12]'. "
    "Answer in as few words as possible — ideally a single value (a place, an "
    "employer, a preference). If the question asks about a specific day, reason "
    "from the most recent message on or before that day. If the question asks "
    "how something changed over time, list the values in chronological order, "
    "comma-separated. If the messages do not say, answer 'unknown'."
)


def build_rag_answer_user_prompt(question: str, snippets: Sequence[str]) -> str:
    """Render the retrieved messages and the question into one user turn."""
    if snippets:
        messages = "\n".join(snippets)
    else:
        messages = "(no messages retrieved)"
    return f"Messages:\n{messages}\n\nQuestion: {question}\nAnswer:"


JUDGE_SYSTEM_PROMPT = (
    "You grade whether a candidate answer matches the reference answer for a "
    "question. Judge meaning, not wording: case, punctuation, and extra words "
    "do not matter, and a place or name matches its obvious variants. For a "
    "question about change over time, every reference value must appear, in the "
    "right order, for the answer to be correct. Respond with a single JSON "
    'object: {"correct": true or false, "reason": "<brief>"}. No other text.'
)


def build_judge_user_prompt(
    question: str, reference: Sequence[str], answer: str
) -> str:
    """Render the question, gold reference, and candidate answer for grading."""
    reference_text = ", ".join(reference) if reference else "(none)"
    return (
        f"Question: {question}\n"
        f"Reference answer: {reference_text}\n"
        f"Candidate answer: {answer}\n"
        "Is the candidate answer correct?"
    )
