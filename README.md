<h1 align="center">
Mneme
</h1> (Μνήμη)
> is an Ancient Greek word meaning: memory, remembrance, or the faculty of memory.

<h2 align="center">A From-First-Principles Architecture for AI Long-Term Memory</h2>

<div>
TL;DR: I'm trying to build a three-layer, append-only "engram store" with a shared substrate, not three databases.
A single immutable, content-addressed event log (Merkle DAG) feeds three co-derived indexes — a quantized vector graph for semantic recall, an Elias-Fano/learned temporal index for exact episodic and time-travel recall, and a bitemporal knowledge graph for causal/belief evolution.
  
All three are projections of one log, so they stay consistent and share storage. Treat "forgetting" as belief revision, never as deletion. Borrow bitemporal (valid-time/transaction-time) modeling plus truth-maintenance/AGM supersession: facts get validity intervals and are invalidated (tombstoned with a successor pointer), not erased. Lossy compression is applied only to superseded content (delta-encoded against its successor), so still-valid information is never silently lost.
  
It is still in the prototype phase building on existing primitives: an LSM/log-structured store, the SDSL succinct-structures library, FAISS/DiskANN or SPANN+SPFresh for the vector layer with RaBitQ 1-bit quantization, Matryoshka-truncatable embeddings, and a Graphiti-style bitemporal graph.

The key parts that are different from other implementations are the consolidation pipeline and the shared-substrate routing, not any new ANN math.

</div>

This is not <a href="https://github.com/getzep/graphiti">Graphiti</a> but it is the nearest thing conceptually and it borrows some ideas from it. [![arXiv](https://img.shields.io/badge/arXiv-2501.13956-b31b1b.svg?style=flat)](https://arxiv.org/abs/2501.13956)

I am not trying to build "another" graph/vector database with memory features. MNEME is an event-sourced memory operating system for AI.

---

## MVP — implemented so far

The MVP is a deliberately small Python/SQLite slice of the architecture above: **one append-only log as the source of truth, and a fact projection that is rebuildable from it.** Everything on the scaling list (Merkle DAG, Elias-Fano/learned temporal index, RaBitQ quantization, ATMS) is intentionally deferred until a measured number justifies it.

The MVP exists to settle **one** question: is supersession-based memory worth it at all? The whole project is measured against the **B0 ablation** — the same pipeline with history-preserving supersession swapped for last-write-wins overwrite. A loss to B0 means supersession was never worth the complexity. The ingest spine (WS1→WS2→WS3), the semantic candidate path (WS4), the read side (WS5), the gold dataset (WS7), and the scoring harness that turns them into the B0 number are in; the remaining baselines (B1/B2/B3) are next.

The B0 gate, run offline against the gold scenarios (`python scripts/eval_harness.py`):

```
system     overall  current  historical  evolution
supersede  100%     100%     100%        100%
overwrite  50%      100%     0%          33%
```

`current` is a tie — overwrite keeps the latest belief, so it answers "where does alice live now?" just fine. The gap is the whole thesis: overwrite scores **0% on `historical`** and collapses to **33% on `evolution`** (only a never-changed fact, a chain of one, survives), because it threw the past away. Supersession keeps it and answers everything.

### Ingest — the write path (WS1–WS4)

```mermaid
flowchart TD
    turn(["conversation turn"]) -->|"EventLog.append"| events[("events<br/>append-only log · source of truth<br/>UPDATE / DELETE locked by triggers")]

    events -->|"replay"| extractor["LLMExtractor · WS2<br/>event → subject, predicate, object, valid_from"]
    extractor -->|"ExtractedFact"| policy{"WritePolicy · WS3<br/>Supersede · Overwrite/B0 · InsertOnly"}

    provider["CandidateProvider · WS4<br/>subject-match or top-k semantic neighbours"]
    facts[("facts<br/>derived read-model · bitemporal")]
    detector["ContradictionDetector · WS3<br/>new / duplicate / refines / supersedes"]

    facts -.->|"candidate set"| provider
    provider -.-> policy
    policy -->|"judge"| detector
    detector -->|"relation"| policy
    policy -->|"insert / close_out"| facts

    subgraph rebuild ["FactStore.rebuild() — facts are re-derivable from the log"]
        extractor
        policy
        detector
    end
```

The invariant that is the whole architecture: **`events` is pure append (enforced in the schema by `UPDATE`/`DELETE` triggers), and `facts` is a projection that `FactStore.rebuild()` can throw away and re-derive from the log at any time.** If extraction improves or the projection is corrupted, you replay the log and rebuild.

### Read — the query path (WS5)

```mermaid
flowchart LR
    facts[("facts<br/>bitemporal projection")] --> router["QueryRouter · WS5"]
    router --> current["current<br/>belief in force now"]
    router --> historical["historical(as_of)<br/>belief at an instant"]
    router --> evolution["evolution<br/>walk superseded_by chain"]

    current -.->|"survives B0"| ok([" ✅ overwrite can answer "])
    historical -.->|"needs history"| gone([" ❌ overwrite lost it "])
    evolution -.->|"needs history"| gone
```

`historical` and `evolution` are the discriminator: overwrite keeps one row per slot, so a past instant resolves to nothing and the evolution chain collapses to length one. That gap is what the eval harness measures (the B0 table above).

### WS1 — schema + append-only event log

- `events` (immutable) and `facts` (derived, bitemporal: valid-time + transaction-time) tables — `mneme/db/schema.sql`.
- `EventLog.append / get / replay` — `mneme/log/event_log.py`.
- `FactStore` — `insert`, `close_out` (the supersession write), `overwrite` (the B0 destructive write), `current_for`, `slot_facts`, `current_facts`, `rebuild` — `mneme/facts/store.py`. `facts` is left mutable on purpose so the Overwrite/B0 ablation can last-write-wins in place.

### WS2 — LLM fact extractor

- `LLMExtractor` turns one event into `(subject, predicate, object, valid_from)` candidates, implementing the `Extractor` seam so it plugs straight into `FactStore.rebuild` and every baseline — `mneme/facts/llm_extractor.py`.
- One shared `LLMClient` / `AnthropicClient` serves both extraction (recall-tuned) and the contradiction judge (precision-tuned) — same model, different operating points — `mneme/llm/`.
- Model output is untrusted: the required triple is parsed strictly and raises `ExtractionError` rather than guessing, while the optional `valid_from` is best-effort — coarse forms (`2026-Q1`, `2026-03`) resolve to the period start and unrecognized dates degrade to the event timestamp with a warning, so one fuzzy date never aborts a run.

### WS3 — contradiction detector + write policies (the thesis and the risk)

- `ContradictionDetector` classifies a candidate against the facts it might touch as `new` / `duplicate` / `refines` / `supersedes`, tuned for precision and short-circuiting to `new` when there is nothing to compare against (no LLM call, no chance to hallucinate a conflict) — `mneme/facts/detector.py`. The judgment is untrusted external data, validated strictly.
- The defining error is a **false supersession** — closing out a fact that was not really contradicted — so it is tracked as its own metric.
- Three policies share one extractor and one store, so the comparison is structural and free — `mneme/facts/policy.py`:
  - `SupersedePolicy` — the thesis: on a conflict, insert the new fact and `close_out` the old one on both temporal axes, keeping full history.
  - `OverwritePolicy` — the **B0 ablation**: last-write-wins on the subject+predicate slot, in place, history gone.
  - `InsertOnlyPolicy` — the no-conflict-handling default used by `rebuild`.

### WS4 — embeddings + FAISS HNSW semantic index

- `FastEmbedEmbeddingClient` — local ONNX embeddings (`BAAI/bge-small-en-v1.5`, 384-dim), **no API key required** — `mneme/embeddings/`.
- `FaissHnswIndex` (cosine via inner product over L2-normalized vectors) wrapped by a domain-agnostic `SemanticIndex` — `mneme/index/`.
- `CandidateProvider` feeds the detector the facts worth comparing against: `SubjectCandidateProvider` (exact subject match) or `SemanticCandidateProvider` (top-k neighbours), so what you embed at fact granularity is load-bearing — `mneme/facts/candidates.py`.

### WS5 — query router

- `QueryRouter.current / historical(as_of) / evolution` over a `(subject, predicate)` slot — `mneme/query/router.py`. Deterministic and LLM-free: it takes a structured slot, not a natural-language question.
- `evolution` walks the `superseded_by` chain from its unique head, with a valid-time fallback and a seen-guard so a corrupted projection can never loop.

### WS7 — synthetic dataset + the B0 eval harness

- Hand-authored, self-checked gold scenarios: known timelines whose facts, supersession relations, and query answers are validated for internal consistency at authoring time — `mneme/eval/dataset.py`, `mneme/eval/validate.py`.
- The gold is simultaneously the **spec for the detector** (each event carries the relation it should be judged as) and the **discriminator for the B0 gate** (its `historical`/`evolution` queries are only answerable by a history-preserving store). `materialize()` renders a scenario into immutable log events.
- `ScenarioOracleDetector` swaps the LLM judge for the gold relations, so the harness runs offline and **deterministically** — holding extraction and judgment fixed so the only variable between the two systems is the storage policy — `mneme/eval/oracle.py`.
- The harness ingests each scenario into a fresh in-memory store under Supersede vs Overwrite, runs the router over every gold query, and scores answer accuracy by query kind — `mneme/eval/harness.py`, run via `scripts/eval_harness.py`. The supersede-minus-overwrite gap on `historical`/`evolution` **is** the B0 result above.

### Run it

```bash
pip install -e '.[dev,vectors,embeddings]'   # core + tests + FAISS + local embeddings
pytest                                         # 160 tests, no API key needed
python scripts/eval_harness.py                 # the B0 gate, offline + keyless

pip install -e '.[llm]'                        # adds the anthropic client
ANTHROPIC_API_KEY=… python scripts/extract_demo.py     # eyeball extraction
ANTHROPIC_API_KEY=… python scripts/supersede_demo.py   # full supersession pipeline
python scripts/semantic_demo.py                        # local embeddings + FAISS, keyless
```

**Next:** WS6 baselines (B1 raw RAG, B2 summary, B3 Graphiti-like). The B0 gate is settled — supersession beats overwrite outright on the history-dependent queries; the remaining baselines test it against the RAG-style alternatives the field actually reaches for.
