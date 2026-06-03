"""Prompts for the RAG-style baselines and the shared judge.

Deliberately plain. B1 (raw RAG) and B2 (running summary) are the straw men the
thesis must beat, so neither gets any notion of supersession or valid-time —
B1's answerer sees only retrieved messages, B2's sees only a compressed running
summary, exactly the systems MNEME claims to improve on. The judge is shared
scoring machinery, kept separate so the same grader scores any baseline's
free-text answer.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "RAG_ANSWER_SYSTEM_PROMPT",
    "SUMMARY_UPDATE_SYSTEM_PROMPT",
    "SUMMARY_ANSWER_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "build_rag_answer_user_prompt",
    "build_summary_update_user_prompt",
    "build_summary_answer_user_prompt",
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


SUMMARY_UPDATE_SYSTEM_PROMPT = (
    "You maintain a concise running summary of what is known about a person from "
    "their chat messages. Given the summary so far and the next message (tagged "
    "with the day it was sent, like '[day 12]'), return an updated summary that "
    "folds in the new message. Keep it to a few sentences. Return only the "
    "updated summary text — no preamble, no commentary."
)


def build_summary_update_user_prompt(summary: str, message: str) -> str:
    """Render the summary-so-far and the next message into one user turn."""
    return (
        f"Summary so far:\n{summary or '(none yet)'}\n\n"
        f"New message: {message}\n\n"
        "Updated summary:"
    )


SUMMARY_ANSWER_SYSTEM_PROMPT = (
    "You answer questions about a person using only the running summary provided. "
    "Answer in as few words as possible — ideally a single value (a place, an "
    "employer, a preference). If the question asks about a specific day, use what "
    "the summary says about that time. If the question asks how something changed "
    "over time, list the values in chronological order, comma-separated. If the "
    "summary does not say, answer 'unknown'."
)


def build_summary_answer_user_prompt(summary: str, question: str) -> str:
    """Render the running summary and the question into one user turn."""
    return f"Summary:\n{summary or '(empty)'}\n\nQuestion: {question}\nAnswer:"


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
