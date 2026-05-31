<h1 align="center">
MENEME
</h1>
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
