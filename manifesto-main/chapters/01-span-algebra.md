> [← Ch.0: Introduction](00-introduction.md) · [Table of Contents](../README.md) · [Ch.2: LLMON →](02-llmon.md)

---

# Chapter 1: The Span Algebra

**Status:** Draft
**Version:** 0.5
**Authors:** David Cox, Claude
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-06

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

This chapter presents the formal foundation shared by all subsequent chapters: the mathematical objects, operations, and composition laws that govern KV cache structure during training, serving, and corpus-scale indexing.

## 1.1 Spans and KV Cache Representations

A **span** is a contiguous region of context — both a token sequence and the corresponding KV cache representation. Spans are named with uppercase letters (A, B, C, ...). The KV cache representation of span A is written **A'**. The concatenation of spans A and B is written **(A ⊕ B)**.

A critical asymmetry: in general, **(A ⊕ B)' ≠ A' ⊕ B'**. The KV cache for B depends on which tokens precede it during prefill, because causal self-attention allows each token to attend to all preceding tokens. Prefilling B after A produces a different B' than prefilling B in isolation — B's key and value vectors reflect its attention over A's tokens. This asymmetry is the source of much of the subtlety, and much of the expressive power, in the span formalism.

## 1.2 Fundamental Operations

**Prefill.** The prefill operator populates the KV cache for a span, respecting causal attention dependencies:

```
A' = fill(A)
```

Because of the asymmetry above, prefilling a concatenation is not equivalent to independently prefilling the parts:

```
fill(A ⊕ B) ≠ fill(A) ⊕ fill(B)
```

When B is prefilled after A (with A's cache in scope), B's cache reflects its attention over A's tokens — a different B' than would arise from prefilling B alone.

**Generation.** The generate operator produces new tokens and their KV cache entries autoregressively, given a scope:

```
B = gen(A)          -- generate B with A in scope
F = gen([M, N, Q])  -- generate F with multiple spans in scope
```

Generation scopes need not be contiguous. Pre-cached instruction or data spans from different sources can be composed freely at generation time.

**Interleaving.** Generation and prefill compose to build multi-step inference pipelines:

```
R1 = gen([D, I])     -- generate summary R1 from data D and instruction I
R2 = gen([C, R1])     -- generate reflection R2 from check prompt C and summary R1
```

Each step's output becomes an input span for subsequent steps. The span abstraction makes dataflow explicit: R2 depends on R1 and C, but not directly on D or I (unless they are included in R2's scope).

## 1.3 Scope Algebra

In conventional LLM inference, scope is implicit: every token attends to every preceding token. The span model makes scope **explicit**. To know what KV cache a span will have, you must know which other spans were in scope when it was populated:

```
B' = fill(B, [A'])    -- B prefilled with A in scope
B' = A' ∘ B           -- equivalently, using the composition operator
A' = ø ∘ A            -- A prefilled with nothing in scope
```

Scopes compose via two operations:

**Union (∪).** A commutative operation that combines spans into a shared scope:

```
C' = (A' ∪ B') ∘ C   -- C prefilled with both A and B in scope
```

**Composition (∘).** A non-commutative operation reflecting sequential dependency introduced by attention. `A' ∘ B` means "B was prefilled with A in scope." Non-commutativity reflects the asymmetry of §1.1: the order of composition determines which tokens attend to which predecessors.

This is isomorphic to dependency-graph algebra and supports formal reasoning about information flow through multi-span computations:

```
A' = ø ∘ A
B' = A' ∘ B = (ø ∘ A) ∘ B
C' = (A' ∪ B') ∘ C
```

## 1.4 Fan-Out and Fan-In

**Fan-out (branching).** Because scopes are explicit, two different instructions can operate on the same cached data without interfering:

```
R1 = gen([D, I1])     -- instruction I1 applied to data D
R2 = gen([D, I2])     -- instruction I2 applied to same data D
```

R1 and R2 share the data span D but diverge based on their respective instructions, with no cross-contamination. Fan-out is already well-supported by prefix caching in existing engines.

**Fan-in (merging).** The complementary operation — independently populated spans joined into a common scope for generation:

```
A' = fill(A)
B' = fill(B)
C = gen([A', B'])     -- C attends to both A and B, but A and B are independent
```

Fan-in requires the inference engine to construct attention masks over non-contiguous cache regions — a capability that the serving engine in Chapter 5 is designed to provide.

Fan-in enables pre-computed reusable cache blocks (RAG passages prefilled once and composed per-query), sensitive information isolation (include in scope only where necessary), and clean argument scopes (arguments untainted by each other, so I(P1, P2) = I(P2, P1) up to the model's content-conditional behavior, not its positional artifacts).

## 1.5 Tainting, Position Invariance, and the Free Monoid

The asymmetry of §1.1 — (A ⊕ B)' ≠ A' ⊕ B' — is a consequence of **attention-mediated tainting**, not of position encoding. When B is prefilled with A in scope, every token in B attends to every token in A via causal self-attention. B's key and value vectors are shaped by this attention — they carry information derived from A. Prefill B in isolation and you get a different B', because the attention patterns are different. This is true regardless of how positions are encoded, or whether they are encoded at all. The scope algebra (§1.3) manages this: by making scope explicit, it makes tainting controllable. Two spans are **independently scoped** when neither was in the other's scope during prefill — each was prefilled in isolation (or under scopes that do not include the other). For independently scoped spans, there is no tainting: their KV caches are pure functions of their own content and their own (disjoint) scopes, and can in principle be composed.

**Position encoding imposes a second, orthogonal constraint.** Under RoPE, each key is rotated by an angle proportional to its absolute position. Even for independently scoped spans (no tainting issue), a KV segment computed at positions 0–99 cannot be reused at positions 500–599 — the keys encode the wrong positions. This constrains composition to segments whose positional provenance forms a compatible chain, a partial order. RoPE doesn't cause the tainting asymmetry; it adds a second barrier to composition on top of it.

**NoPE removes the positional constraint.** Models without positional encoding in their attention layers produce pure content projections: K = x @ W_k, no position-dependent rotation. For independently scoped spans, the tainting constraint is already absent (they were prefilled without each other in scope) and the positional constraint is now also absent. Both barriers to composition are gone, and the set of independently scoped KV cache segments under concatenation forms a **free monoid**:

**Associativity.** `compose(A, compose(B, C)) = compose(compose(A, B), C)`.

**Identity.** The empty span ε is the identity: `compose(A, ε) = compose(ε, A) = A`.

**No ordering constraint on content.** For independently scoped spans, `compose(A, B)` and `compose(B, A)` produce identical attention output under NoPE.

**Granite** is the primary target for this property. In its hybrid SSM-attention configuration, Granite interleaves Mamba-2 layers with attention layers. The SSM layers maintain a fixed-size hidden state that captures sequential structure implicitly — they do not produce KV caches. The attention layers produce KV caches for precise retrieval, and because they use NoPE, these caches are pure functions of span content. Positional awareness comes from the SSM layers, not from the key projections.

This is the structural precondition for everything in Chapters 4–6: span parallelism in training, zero-compute composition in serving, and offline indexing at corpus scale.

**DRoPE retrofit** provides a migration path for existing RoPE-trained models. Gelberg et al. (2026, Sakana AI) showed that RoPE can be treated as a training scaffold: drop positional embeddings and recalibrate at ~0.5% of pretraining cost. For Granite specifically, DRoPE bridges existing RoPE-trained variants to a position-invariant regime without full retraining.

## 1.6 Tagging and Boundary Types

The **tag** operation labels a span with a semantic role: `tag(span, label, [value])`. Tags can be system-defined (INSTRUCT, DATA, EXEC, SENSITIVE) or user-defined.

**Painting** is the general tagging mechanism: a function applied to token embeddings within a span. Special tokens are a special case (the embedding is overridden entirely). Position embeddings are a special case (each token is painted with a positional tag).

**Wrapping** — delimiting spans with special tokens — is the dominant tagging mechanism in practice. A model trained on wrapped spans learns a form of painting implicitly: the opening delimiter shifts the model into a state that persists through subsequent tokens' key representations until the closing delimiter resets it.

LLMON (Chapter 2) provides the concrete markup language that realizes tagging as a minimal set of special tokens in the model's token stream.

## 1.7 Provenance and Taint

A span's scope record — the set of spans whose KV was visible during its computation — is **provenance metadata**. It answers: "what information could have influenced these cached key-value pairs?"

Provenance matters because attention is an information channel. If span B's KV was computed while attending to span A, then B's cached values carry information derived from A, even if B's tokens make no reference to A. The **refill** operation uses provenance to break taint: discard a span's KV, recompute under a new scope (typically empty), update the provenance record. Under NoPE, the recomputed KV is valid at any slot position — there is no need to match the original positions.

## 1.8 Summary of Algebraic Properties

| Property | Statement | Consequence |
|---|---|---|
| Tainting asymmetry (universal) | (A ⊕ B)' ≠ A' ⊕ B' — prefilling B after A produces different KV than prefilling B alone | Composition of scoped spans requires explicit scope tracking |
| Scope algebra | Union (∪) combines spans; composition (∘) imposes sequential dependency | Makes tainting explicit and controllable; enables fan-in and fan-out |
| Positional entanglement (RoPE) | Keys encode absolute position; reuse requires positional compatibility | Second constraint on composition, even for independently scoped spans |
| Position invariance (NoPE) | KV = f(tokens, scope), independent of position | Removes the positional constraint; independently scoped spans compose freely |
| Free monoid (NoPE) | Independently scoped spans compose by unrestricted concatenation | Span parallelism in training; free composition in serving; offline indexing |
| Provenance tracking | Each span records the scope under which its KV was computed | Taint breaking via refill; verifiable information flow |

Two constraints govern composition: **tainting** (managed by the scope algebra — make scope explicit, compose only independently scoped spans) and **positional entanglement** (eliminated by NoPE). Under Granite's NoPE attention layers, both constraints are resolved for independently scoped spans → **free monoid → {span parallelism, free composition, offline indexing} → the systems in Chapters 4–6**.

---

> [← Ch.0: Introduction](00-introduction.md) · [Table of Contents](../README.md) · [Ch.2: LLMON →](02-llmon.md)
