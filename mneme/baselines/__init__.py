"""Baselines the supersession thesis is measured against (workstream 6).

B0 (overwrite) lives in the B0 gate (``mneme.eval.harness``) because it shares
MNEME's structured storage. This package holds the RAG-style baselines that do
not — they answer in free text and are graded by an LLM judge:

  * ``RawRagBaseline`` (B1) — embed every message, retrieve the nearest, let the
    model answer from the raw text. No facts, no history.
  * ``SummaryBaseline`` (B2) — fold every message into one running summary, then
    answer from that summary alone. Lossy compression, no raw recall.

B3 is the odd one out: a faithful bitemporal store with exact structured answers,
so it is scored on the B0 gate's exact-match path (``SystemReport``), not the
LLM-judge path the free-text baselines need — see ``bitemporal``.
"""

from mneme.baselines.bitemporal import (
    B3_BITEMPORAL,
    BitemporalEdge,
    BitemporalStore,
    evaluate_bitemporal,
)
from mneme.baselines.harness import (
    B1_RAW_RAG,
    B2_SUMMARY,
    BaselineOutcome,
    BaselineReport,
    evaluate_rag,
    evaluate_summary,
    format_baselines,
)
from mneme.baselines.judge import JudgeError, LLMJudge, Verdict
from mneme.baselines.rag import DEFAULT_TOP_K, RawRagBaseline
from mneme.baselines.summary import SummaryBaseline

__all__ = [
    "B1_RAW_RAG",
    "B2_SUMMARY",
    "B3_BITEMPORAL",
    "BaselineOutcome",
    "BaselineReport",
    "BitemporalEdge",
    "BitemporalStore",
    "evaluate_bitemporal",
    "evaluate_rag",
    "evaluate_summary",
    "format_baselines",
    "JudgeError",
    "LLMJudge",
    "Verdict",
    "DEFAULT_TOP_K",
    "RawRagBaseline",
    "SummaryBaseline",
]
