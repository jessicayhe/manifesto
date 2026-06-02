> [← Ch.2: LLMON](02-llmon.md) · [Table of Contents](../README.md) · [Ch.4: Cache-Materialized Training →](04-cache-materialized-training.md)

---

# Chapter 3: Model-Runtime Codesign

**Status:** Draft
**Version:** 0.5
**Authors:** David Cox, Claude
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-06

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

Generative computing is not something a model does alone. It is a property of a **model and runtime designed together**. A model trained on LLMON-structured data learns to respect span boundaries, but the structural guarantees — scope isolation, taint tracking, boundary-constrained masking — are enforced by the runtime, not by the model's learned behavior. Conversely, a runtime that manages spans and enforces scopes is most effective when the model has been trained to produce KV caches that compose well across span boundaries.

This chapter frames the codesign relationship, introduces `mellea-core` as the practical implementation of the runtime side, and identifies the integration surfaces between the three `mellea-core` systems described in Chapters 4–6.

## 3.1 The Codesign Principle

A guiding principle of generative computing: **models and inference-time programs should be codesigned**. The style and domain of prompting used at inference time should match what the model was trained on, and models should be built with runtime components and usage patterns in mind.

This principle has concrete consequences for the systems in this document:

**Training shapes the model for the runtime.** Cache-materialized training (Chapter 4) processes sequences as ordered spans, explicitly materializing KV caches at span boundaries. The model learns to produce KV projections that are useful when composed — not because it is told to via a loss function, but because the training regime itself operates in the compose-and-attend mode that the runtime will use at serving time. The training regime *is* the serving regime, with gradients turned on.

**The runtime enforces what the model cannot guarantee.** LLMON-structured training raises distractor robustness to ~83%, but that leaves ~17% where the model still follows the wrong instruction. The runtime's boundary-constrained masking eliminates that residual by making distractor instructions invisible at the attention level — a structural guarantee that does not depend on what the model learned. Defense in depth: training provides the first line; the runtime provides the backstop.

**Codesign is incremental.** A model trained without LLMON can still be served by the `mellea-core` runtime — the runtime approximates span structure via prompt assembly and provides whatever enforcement the backend supports. A model trained with LLMON but served on a basic API backend still benefits from its learned span semantics. Full generative computing emerges when both sides are designed for each other, but either side provides value alone.

## 3.2 `mellea-core`: The Runtime Implementation

`mellea-core` is the Rust implementation of the generative computing runtime. It provides:

**Span algebra primitives.** `fill`, `gen`, `compose`, `refill` — the operations defined in Chapter 1, implemented over a KV pool with slot-level memory management. Composition is integer list concatenation under NoPE; scope enforcement is boundary-constrained masking derived from LLMON structure or caller-declared scope.

**LLMON parsing.** Tokenization-aware parsing of Machine LLMON special tokens into span boundaries, dependency DAGs, and exec bindings. Feeds both the training pipeline (span boundary extraction for cache-materialized training) and the serving engine (scope mask construction).

**Serving engine.** KV pool with slot allocator, span registry with provenance tracking, batched decode with cascade attention, scope-isolated masking, and the refill (taint-breaking) primitive. Chapter 5 describes this in detail.

**Corpus index.** Hierarchical centroid tree over span KV caches, navigated via stick-breaking attention for O(k log n) retrieval over trillion-token corpora. Chapter 6 describes this in detail.

The implementation targets are Rust (core), PyO3 (Python bindings for training integration and `mellea.stdlib`), and WASM via wasm-bindgen (browser-side `mellea.js`).

## 3.3 The Abstraction Boundary

Every construct in generative computing can be implemented at two levels:

**Native implementation** via `mellea-core`. Span boundaries are enforced at the attention level. Scope isolation is exact (boundary-constrained masking). Composition is free-monoid concatenation. Taint tracking is metadata-precise. The model was trained with LLMON structure and operates within a runtime that enforces it.

**Prompt-assembly approximation** via any backend. Span structure is encoded as text within a flat prompt string. The model must infer boundaries from formatting conventions. Scope isolation is statistical. Composition requires re-prefill. Taint tracking is not available.

The `mellea.stdlib` API is the same in both cases — the abstraction boundary between interface and implementation is what makes progressive adoption possible. Chapter 7 (§7.7) describes how this boundary operates across specific backends and what each level of the stack provides.

## 3.4 The Three Layers

The three `mellea-core` systems described in Chapters 4–6 form a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Corpus Index (Chapter 6 — Span Attention)     │
│    Multilevel centroid tree over the full KV corpus      │
│    Stick-breaking descent selects the working set        │
│    Tiered storage: GPU → host → SSD → object store      │
│    Scale: 10⁶ – 10¹² tokens                             │
└────────────────────────┬────────────────────────────────┘
                         │ materializes selected spans into ↓
┌────────────────────────▼────────────────────────────────┐
│  Layer 2: Serving Engine (Chapter 5)                    │
│    KV pool with slot allocator                          │
│    Span registry with scope, provenance, ref counting   │
│    Batched decode with cascade attention                 │
│    Scale: 10³ – 10⁵ tokens (GPU-resident working set)   │
└────────────────────────┬────────────────────────────────┘
                         │ runs a model produced by ↓
┌────────────────────────▼────────────────────────────────┐
│  Layer 1: Training (Chapter 4)                          │
│    Cache-materialized pretraining (Phase 2)             │
│    Span-sequential forward pass with gradient flow      │
│    Optional learned compression / eviction               │
│    Output: a model whose KV projections are cache-aware  │
└─────────────────────────────────────────────────────────┘
```

Each layer depends on the one below it and extends the one above it. Training produces a model. The serving engine runs that model. The corpus index extends the serving engine to contexts that exceed GPU memory.

In Granite's hybrid configuration, the SSM layers do not produce KV caches — they maintain a fixed-size hidden state updated in-place. Only the attention layers produce KV caches, and only those caches are managed by the serving engine (Layer 2) and indexed by the corpus index (Layer 3). In a typical Granite hybrid configuration, the majority of layers are SSM with a minority of attention layers (e.g., 4 attention layers in a 32-layer model), significantly reducing per-span cache size and making corpus-scale indexing more practical.

## 3.5 Integration Surface: KV Pool as Level 0

The span attention chapter (Chapter 6) defines a hierarchy of abstraction levels. The serving engine's KV pool IS Level 0 — the physical storage of individual K/V pairs. The span attention's gating pass selects which Level 1 spans to materialize; materialization means allocating slots in the KV pool.

**Downward (index → pool).** When the stick-breaking descent selects a span, the serving engine's KV pool must have its KV GPU-resident. If `GpuResident`, gather directly. If `HostResident` or lower, promote via DMA or re-prefill.

**Upward (pool → index).** When the serving engine fills a new span, the span attention index is updated asynchronously: compute Level 1 centroids, insert, propagate summaries upward.

## 3.6 Integration Surface: Scope Resolution at Two Scales

**Serving engine scope** is per-request, per-span. The caller declares visible spans; the engine constructs an attention mask — boundary-constrained masking (Chapter 2, §2.5) applied at the KV pool level. Operates over the GPU-resident working set.

**Span attention scope** is per-query-position, over the full corpus index. Scope filtering happens at Level 1 — upper levels are scope-agnostic structural groupings; the scope check occurs at leaf nodes with boundary metadata.

Span attention scope determines which spans *could* be materialized. Serving engine scope determines which *materialized* spans are visible to a given request. At conversational scale (all KV fits in the pool), only the serving engine's scope resolution is needed. The span attention layer activates when context exceeds GPU memory.

## 3.7 Integration Surface: Algebra to Runtime Primitives

| Algebra Operation | Training (Ch.4) | Serving (Ch.5) | Scaling (Ch.6) |
|---|---|---|---|
| `fill(A)` | `forward_span()` | `fill()` API | Offline pipeline Stage 2 |
| `gen([...])` | N/A | `gen()` / `gen_parallel()` API | N/A |
| `A' ∘ B` (composition) | Cache concatenation in compute graph | `compose()` — integer list concat | Free monoid composition |
| `A' ∪ B'` (scope union) | DAG-based span parallelism | Cascade attention — shared prefix broadcast | Scope filtering at Level 1 |
| `tag(span, label)` | `span_types` in data format | `Span.tag` field | Boundary type at Level 1 |
| `refill(A)` | N/A | `refill()` API — taint break | Re-prefill on cache fault |
| Boundary-constrained masking | Implicit in span-sequential processing | `build_scope_mask()` | Scope check at Level 1 |

A single formal framework governs what the model learns (training), how it runs (serving), and how it scales (indexing). This is the codesign: the algebra is the same; the implementation varies by layer.

## 3.8 What Training Owes Serving

Cache-materialized training (Chapter 4) produces a model checkpoint. The relevant interface to downstream `mellea-core` systems is a set of **behavioral properties** of the trained model's KV projections:

**Composition fidelity.** Cache entries composed via concatenation across span boundaries yield attention outputs close to full-sequence attention. The primary success metric of cache-materialized training and the correctness assumption of the serving engine's compose.

**Clusterability.** Keys within semantically coherent spans should cluster tightly in angular space for effective centroid summarization (Chapter 6). Cache-materialized training does not explicitly optimize for this, but it is a testable property. If clusterability is poor, the span attention index's gating accuracy degrades, and an auxiliary training signal may be warranted.

**Stable norms.** Key vector norms serve as a potency signal in the span attention index (Chapter 6, §2.3). Pathological norm distributions degrade gating calibration.

These are not API contracts — they are empirical properties that must be measured after training and that inform the design of both the training curriculum and the downstream systems. The codesign runs in both directions.

## 3.9 Reading Order

**Model trainer:** Chapter 1 → Chapter 2 (§2.6–2.7) → Chapter 4.

**Serving engineer:** Chapter 1 → Chapter 5 → Chapter 3 §3.5–3.6.

**Retrieval / RAG researcher:** Chapter 1 → Chapter 6 → Chapter 3 §3.5.

**Application developer:** Chapter 7 → Chapter 8 → application chapter (Chapter 10 for conversation, Chapter 11 for reasoning).

**System architect:** All chapters in order.

---

> [← Ch.2: LLMON](02-llmon.md) · [Table of Contents](../README.md) · [Ch.4: Cache-Materialized Training →](04-cache-materialized-training.md)
