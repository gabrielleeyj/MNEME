#!/usr/bin/env python3
"""Run the B1 raw-RAG baseline over the gold scenarios — the live number.

Unlike the B0 gate (offline, exact match), B1 answers in free text and is graded
by an LLM judge, so this needs your Anthropic key. Embeddings run locally via
fastembed (no extra key):

    ANTHROPIC_API_KEY=sk-... python scripts/baselines_demo.py

Install extras first:  pip install -e '.[llm,vectors,embeddings]'

Print the per-kind table so B1 can be read next to the B0 gate: B1's weakness is
meant to show up on `historical` and `evolution`, the queries a store with no
supersession and no valid-time index cannot reliably answer.
"""

from __future__ import annotations

from mneme.baselines.harness import B1_RAW_RAG, evaluate_rag, format_baselines
from mneme.embeddings.client import FastEmbedEmbeddingClient
from mneme.llm.client import AnthropicClient


def main() -> None:
    embedder = FastEmbedEmbeddingClient()
    client = AnthropicClient()
    report = evaluate_rag(embedder, client)
    print(format_baselines({B1_RAW_RAG: report}))


if __name__ == "__main__":
    main()
