#!/usr/bin/env python3
"""Run the RAG-style baselines (B1 raw RAG, B2 summary) over the gold — the live
numbers.

Unlike the B0 gate (offline, exact match), these answer in free text and are
graded by an LLM judge, so this needs your Anthropic key. B1's embeddings run
locally via fastembed (no extra key); B2 keeps no raw messages, so it needs no
embedder:

    ANTHROPIC_API_KEY=sk-... python scripts/baselines_demo.py

Install extras first:  pip install -e '.[llm,vectors,embeddings]'

Print the per-kind table so the baselines can be read next to the B0 gate: their
weakness is meant to show up on `historical` and `evolution`, the queries a store
with no supersession and no valid-time index cannot reliably answer — B1 because
top-k retrieval blurs the timeline, B2 because the running summary drops
superseded detail.
"""

from __future__ import annotations

from mneme.baselines.harness import (
    B1_RAW_RAG,
    B2_SUMMARY,
    evaluate_rag,
    evaluate_summary,
    format_baselines,
)
from mneme.embeddings.client import FastEmbedEmbeddingClient
from mneme.llm.client import AnthropicClient


def main() -> None:
    embedder = FastEmbedEmbeddingClient()
    client = AnthropicClient()
    reports = {
        B1_RAW_RAG: evaluate_rag(embedder, client),
        B2_SUMMARY: evaluate_summary(client),
    }
    print(format_baselines(reports))


if __name__ == "__main__":
    main()
