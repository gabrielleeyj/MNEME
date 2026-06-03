"""Baselines the supersession thesis is measured against (workstream 6).

B0 (overwrite) lives in the B0 gate (``mneme.eval.harness``) because it shares
MNEME's structured storage. This package holds the RAG-style baselines that do
not — they answer in free text and are graded by an LLM judge:

  * ``RawRagBaseline`` (B1) — embed every message, retrieve the nearest, let the
    model answer from the raw text. No facts, no history.
"""

from mneme.baselines.harness import (
    B1_RAW_RAG,
    BaselineOutcome,
    BaselineReport,
    evaluate_rag,
    format_baselines,
)
from mneme.baselines.judge import JudgeError, LLMJudge, Verdict
from mneme.baselines.rag import DEFAULT_TOP_K, RawRagBaseline

__all__ = [
    "B1_RAW_RAG",
    "BaselineOutcome",
    "BaselineReport",
    "evaluate_rag",
    "format_baselines",
    "JudgeError",
    "LLMJudge",
    "Verdict",
    "DEFAULT_TOP_K",
    "RawRagBaseline",
]
