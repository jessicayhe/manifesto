I 9> [← Ch.5: Serving Engine](05-serving-engine.md) · [Table of Contents](../README.md) · [Ch.7: Mellea Standard Library →](07-mellea-stdlib.md)

---

# Chapter 6: Span Attention (Proposal)

**Component:** `mellea-core` / Span Store
**Status:** Draft
**Version:** 0.4
**Authors:** David Cox, Nima Dehmamy, Daniel Weidele
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-06
**Scope:** Extends the `mellea-core` serving engine (Chapter 5) to corpus-scale contexts (10⁶–10¹² tokens).
**Prerequisites:** Chapter 1 (Span Algebra — free monoid, position invariance), Chapter 3 (Model-Runtime Codesign §3.5 — KV pool as Level 0)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---


## 6.0 Purpose

This section describes a proposed new attention mechanism designed to leverage the "span" construct to enable access very large contexts (millions-to-billions of tokens). The opportunity that this design exploits comes from a few unique properties of spans as described in previous sections:

- Spans represent sequential, coherent unit of cache. Spans can corresponds to existing structuring elements of conventional LM contexts, such as turns, tool definitions, tool calls, etc., but can also be used to represent any structured object directly. The notion of span scope allows tree-structured attention patterns where tokens attend with surgical precision to relevant content, and metadata so that those span scopes can be reassembled later, and so that data tainting can be tracked and taints can be broken in a systematic way if desired.
- In the NoPE setting described in this document, span composition is a free monoid, meaning RoPE encodings and the traces of RoPE encodings do not contaminate or complicate the aggregation of relevant content. Assembling a context made up of discontiguous spans is a free operation.
- Span algebra and cache-materialized training admit operations that sort, merge, compress, filter, and otherwise transform span content.
- Spans provide a natural granularity at which to index cache, enabling the possibility of subquadratic attention.

---

## 6.1 Background

### 6.1.0 Glossary of symbols used in this doucment

| Symbol | Definition |
|---|---|
| `n` | Total tokens in the indexed context |
| `d` | Head dimension (typically 64–128) |
| `h` | Number of KV heads (= num attention heads for MHA; fewer for GQA/MQA) |
| `S` | Number of spans |
| `C` | Average number of centroids per span |
| `b` | Branching factor (children per tree node) |
| `k` | Effective branching factor (children visited before stick exhaustion; k ≤ b) |
| `L` | Tree depth; `L = ⌈log_b(n)⌉` |
| `S*` | Selected span set — spans that survive gating for fine attention |
| `\|S*\|` | Total tokens across all spans in S* |
| `ρ` | Potency — aggregate attention-drawing power of a centroid cluster |
| `σ²` | Angular variance of keys within a centroid cluster |
| `β` | Break fraction — the proportion of remaining stick allocated at each step |
| `σ(·)` | Sigmoid function; maps logit to break fraction in `(0, 1)` |
| `τ_open` | Opening threshold — minimum stick allocation to justify descending |
| `ε` | Termination threshold — remaining stick below which traversal stops |

---

### 6.1.1 Mellea and the Span Algebra

**Mellea** is a framework for generative computing built on the concept of **spans** — contiguous segments of a token sequence that carry boundary metadata and compose according to algebraic rules. The **span algebra** defines how spans are concatenated, nested, and scoped: each span has a boundary type that determines which other spans its tokens may attend to during attention computation. This is called **boundary-constrained masking** — an attention mask that is derived from span boundaries rather than from a fixed positional pattern (like causal or sliding-window masks).

The span algebra supports a **scope** mechanism: at any query position, the set of spans whose keys are "in scope" (i.e., eligible for attention) is determined by the nesting and boundary structure, not by absolute position. Scope resolution is a metadata operation — no tensor computation is needed.

### 6.1.2 Position Invariance: Hybrid Architectures and DroPE

A critical property for this design is **position invariance**: the KV cache of a span should not depend on where the span appears in a composed sequence. Standard positional encodings like RoPE (Rotary Position Embedding) violate this — they bake absolute or relative positions into key projections, making KV caches position-dependent and non-composable.

In practice, there are no scaled NoPE-only (No Positional Embedding) models. The practical path to position-invariant KV caches runs through **hybrid SSM-attention architectures** — models that interleave state-space model layers with attention layers (e.g., Jamba, Bamba, Samba, Nemotron-H). In these hybrids:

- **SSM layers** maintain a fixed-size hidden state that captures sequential structure implicitly. They do not produce KV caches and are not indexed by this design.
- **Attention layers** perform precise retrieval via key-value attention. They produce KV caches, and these are the layers this design indexes.

If the attention layers use position-invariant key projections, their KV caches become composable. Two approaches achieve this:

- **NoPE attention layers within a hybrid.** The SSM layers provide implicit positional awareness, so the attention layers can operate without explicit positional encodings. The Samba-NoPE variant (Ren et al., 2024; ICLR 2025) demonstrated this configuration explicitly.
- **DroPE** (Gelberg et al., 2026; Sakana AI): pretrain with RoPE, then drop positional embeddings from the attention layers and run a short recalibration at the original context length. This recovers RoPE-level perplexity while gaining length generalization — treating positional embeddings as a training scaffold that is removed at deployment. DroPE provides a practical retrofit path for existing RoPE-trained models.

Under either approach, the attention layers' KV caches are pure functions of span content, computed once and valid under any composition. Formally, spans compose as a **free monoid** — concatenation is associative, there is an identity (the empty span), and there are no commutativity or ordering constraints imposed by the representation. This is what makes offline precomputation (§6) and hierarchical indexing (§4.7) possible.

Note that stick-breaking attention (§4.1) is itself a natural fit for hybrid architectures — Tan et al. (2024) showed that each stick-breaking head is equivalent to a selective SSM, so it already bridges the two paradigms.

### 6.1.3 FlashAttention-3 and Its Constraints

FlashAttention-3 (Shah et al., 2024) is the target fine-grained attention kernel for this design. FA-3 achieves 1.5–2× speedup over FA-2 on H100 GPUs via three Hopper-specific techniques: warp-specialized overlap of compute and memory transfer, interleaved matmul/softmax, and FP8 block quantization with incoherent processing.

However, FA-3's design assumes properties that span-composed KV caches may violate:

- **Contiguous memory layout.** FA-3's TMA (Tensor Memory Accelerator) handles async data transfer for contiguous tiles. Non-contiguous spans from different origins break this assumption.
- **Uniform FP8 quantization.** FA-3's FP8 path applies per-block scaling factors. Spans quantized independently have mismatched scaling factors that cannot be naively concatenated.
- **Regular mask topology.** FA-3 optimizes for simple, predictable mask patterns (e.g., dense or triangular) that allow entire tiles to be skipped. Boundary-constrained masking can produce irregular block-structured patterns. However, after gating, the selected span set typically has a simple mask — often fully dense (all selected spans are in-scope for the query), which is the easiest case for FA-3.
- **Global softmax normalization.** Online softmax accumulates statistics across the full sequence. When the relevant span set varies dynamically, the "full sequence" is not known until after gating.

This design resolves each of these constraints by ensuring that FA-3 only ever sees a pre-compacted, uniformly-quantized, simply-masked subset of spans selected by the hierarchical gating pass (§7).

## 6.2 Span Summary Representation

### 6.2.1 Design Principle: Norms Carry Information

In standard attention, key vector norms modulate retrievability — a key with large norm produces larger pre-softmax dot products with all queries, capturing more attention weight. Normalizing keys to the unit sphere discards this signal. The summary representation must preserve both **direction** (what the span is about) and **potency** (how strongly it attracts attention).

This parallels the gravitational N-body case: the Barnes-Hut centroid carries both a position and a total mass. A cluster of many faint keys and a single loud key at the same location have very different effects; the summary must distinguish them.

### 6.2.2 The `SpanSummary<M: Metric>` Trait

The summary representation is parameterized by a metric type, allowing progressive sophistication:

```
trait Metric {
    type Centroid;
    type Params;
    fn fit(keys: &KeyMatrix, config: &Params) -> Vec<Self::Centroid>;
    fn logit(query: &QueryVec, centroid: &Self::Centroid) -> f32;
}
```

Note: the trait exposes `logit` (a raw scalar) rather than a normalized score. The stick-breaking process (§4) consumes the logit via a sigmoid to produce the break fraction. This separates the metric geometry from the attention budget mechanism.

The same trait is used at every level of the hierarchy (§4.7). A Level `l+1` centroid is fit over the Level `l` centroid directions below it — the centroid-of-centroids is itself a centroid. The `KeyMatrix` input to `fit` is the matrix of child centroid directions at the level below (at Level 1, this is the span's raw key matrix; at Level 2+, it is the matrix of Level l centroid directions).

Three metric variants, in order of increasing fidelity:

#### `UnitSphere` (vMF)

- **Centroid:** unit direction vector `d ∈ S^{d-1}`
- **Fit:** normalize keys, run spherical k-means (or vMF mixture EM)
- **Logit:** `z = q · d`
- **Use case:** architectures that already normalize keys (e.g., nGPT-style)
- **Storage per centroid:** `d` floats (head_dim, typically 64–128)

#### `ScaledSphere` (direction + potency) — **recommended default**

- **Centroid:** `(d, ρ, σ²)` where `d` is unit direction, `ρ` is aggregate potency (see §2.3), and `σ²` is angular variance of the cluster
- **Fit:** cluster in full key space, then decompose each cluster mean into direction + norm; compute variance
- **Logit:** `z = ρ · (q · d)`, with `σ²` available to the gating decision (§4.2) for coherence-aware opening
- **Use case:** general-purpose; preserves norm information without full covariance
- **Storage per centroid:** `d + 2` floats

#### `Euclidean` (full Gaussian)

- **Centroid:** `(μ, Σ)` — full mean and covariance (or diagonal covariance as a practical compromise)
- **Fit:** standard Gaussian mixture EM in key space
- **Logit:** negative Mahalanobis distance (or log-likelihood under the Gaussian)
- **Use case:** when the direction × scalar factorization is too lossy (empirical determination needed)
- **Storage per centroid:** `d + d` floats (diagonal) or `d + d(d+1)/2` (full)

### 6.2.3 Potency

Potency `ρ` summarizes the attention-drawing power of a cluster. Several definitions are possible; the choice affects gating behavior:

| Definition | Formula | Semantics |
|---|---|---|
| Mean norm | `ρ = (1/n) Σ ‖k_i‖` | Average loudness |
| Sum norm | `ρ = Σ ‖k_i‖` | Total mass (Barnes-Hut analog) |
| Max norm | `ρ = max ‖k_i‖` | Worst-case loudness |
| Norm of mean | `ρ = ‖(1/n) Σ k_i‖` | Coherence-weighted; low if keys point in diverse directions |

**Recommendation:** `sum norm` for the gating decision (total-mass analogy), with `norm of mean / mean norm` stored as a **coherence ratio** that indicates how well the centroid represents its cluster. Low coherence → the centroid is a poor summary → bias toward opening (see §4.2).

**Potency aggregation across tree levels.** At Level `l+1`, the potency of a node is computed by **explicit summation** of its children's potencies at Level `l`, not derived from the `fit` method. This preserves the total-mass invariant: the root-level potency equals the sum of all leaf-level potencies. The `fit` method determines the centroid *direction*; potency is aggregated separately. This distinction is important because fit-derived potency (the norm of the cluster mean) conflates direction coherence with mass — a cluster of diverse high-norm keys would have low fit-derived potency but high sum potency.

### 6.2.4 Summary Values

The key centroid determines relevance; the **value summary** determines the approximate output contribution when a node is used at the coarse level without descending further.

Two modes:

- **Hard gating (skip).** Nodes whose stick allocation falls below the opening threshold contribute nothing. No value summary needed. Simpler, but introduces a discontinuity.
- **Soft gating (approximate + refine).** All nodes contribute — those below the opening threshold contribute via their summary value (weighted by their stick allocation); those above are opened for finer resolution. Requires a value summary per centroid.

For soft gating, the value summary for centroid `j` is:

```
v_j = Σ_i∈cluster_j  α_i · v_i
```

where `α_i` are intra-cluster attention weights precomputed at store time using the centroid direction `d_j` as a proxy query:

```
α_i = softmax_over_cluster( (d_j · k_i) / √d_head )
```

This approximates what a query aligned with the centroid direction would retrieve from the cluster. It is a **weighted mean value** biased toward the keys most aligned with the centroid.

The stick-breaking remainder (§4.3) provides a natural "none of the above" signal. Under soft gating, the remainder's contribution can be mapped to a learned default value vector or to zero (equivalent to the model deciding the context has nothing useful to add).

**Soft gating and the multilevel tree.** Under soft gating in the multilevel hierarchy (§4.7), a node that receives stick budget but is not opened contributes its summary value at the allocated weight. This is the FMM's "far-field approximation" — distant clusters contribute via their multipole expansion rather than their individual particles. Only nodes where the approximation is insufficient (high stick allocation + low coherence) are opened for finer resolution.

At upper tree levels, value summaries are computed recursively: a Level `l+1` node's value summary is a weighted combination of its Level `l` children's value summaries, using the children's potencies as weights:

```
v_parent = Σ_j (ρ_j / Σ_j ρ_j) · v_child_j
```

This recursive composition introduces compounding approximation error — each level of summarization loses fidelity. The quality of the far-field approximation degrades with tree depth, which is an additional reason (beyond compute cost) to prefer opening high-allocation nodes rather than relying on their summaries.

**Recommendation:** Start with hard gating for simplicity. The soft gating extension is backward-compatible (adds an optional field to the summary) and becomes increasingly valuable at higher tree levels where the approximation covers more tokens.

### 6.2.5 Adaptive Expansion Order

The number of centroids per span should scale with the span's internal diversity:

```
n_centroids = max(1, round(H(keys) / H_threshold))
```

where `H(keys)` is some measure of the span's key-space entropy (e.g., effective rank of the key matrix, or the BIC/AIC-optimal number of mixture components). Short, coherent spans (a single sentence on one topic) get one centroid. Long, heterogeneous spans (a full document with multiple sections) get several.

**Upper bound:** Cap at `k_max` centroids per span (e.g., 8–16) to keep the index size bounded. If a span needs more, it should probably be split into sub-spans — and in the multilevel hierarchy, this splitting happens naturally via the tree structure.

## 6.3 Index Structure

### 6.3.1 Layout

Each level of the index tree stores its entries as a **contiguous, sorted array**. The per-level structure is:

```
LevelIndex {
    // Dense, contiguous arrays — one entry per node at this level
    directions: Tensor,       // [n_nodes, head_dim] — unit vectors
    potencies: Tensor,        // [n_nodes] — scalar ρ (sum of children's potencies)
    variances: Tensor,        // [n_nodes] — scalar σ²
    coherences: Tensor,       // [n_nodes] — coherence ratio

    // Per-node metadata
    node_ids: Vec<NodeId>,    // unique identifier for each node
    value_summaries: Option<Tensor>,  // [n_nodes, head_dim] — for soft gating
    child_ranges: Vec<Range<usize>>,  // range into the level below

    // Leaf-level only (Level 1)
    span_lengths: Option<Vec<usize>>,        // tokens in the full span
    cache_status: Option<Vec<CacheStatus>>,  // residency state (§5.2)
    quant_params: Option<Vec<QuantParams>>,  // per-span FP8 scaling factors

    // Index-level metadata
    sort_order: SortOrder,
}
```

The full tree is:

```
IndexTree {
    levels: Vec<LevelIndex>,  // levels[0] = Level 1 (span centroids)
                               // levels[L-1] = root
}
```

The generation counter for double-buffering lives on `SpanStore` (§5.6), not on individual `LevelIndex` instances — a swap replaces the entire tree atomically.

Fields like `span_lengths`, `cache_status`, and `quant_params` are only populated at Level 1 (the leaf level of the index, whose children are the raw KV caches at Level 0). Upper levels have only the centroid data and child pointers.

### 6.3.2 Sort Order

Within each level, entries are grouped by parent and sorted by **potency descending within each parent's child range**. In the single-level case (§4.2), there is no parent grouping and the sort is global across the entire level. In the multilevel case, `child_ranges` in the parent level point to contiguous, potency-sorted slices of the child level — this ensures that the `descend()` traversal (§4.7) encounters the highest-potency children first within each node.

Under stick-breaking, this sort order has a specific semantic role: the stick is consumed in traversal order, so high-potency children get first access to the parent's allocated budget. This creates a natural priority system — the loudest children are evaluated first, and quieter children only receive attention budget if the loud ones didn't consume it.

This is analogous to the Barnes-Hut traversal order: process the largest/nearest clusters first, and only open smaller/distant ones if the approximation error (remaining stick) warrants it.

Alternative sort orders to explore:

- **Key-space locality** (angular proximity via space-filling curve on the hypersphere): better memory access patterns when opening adjacent clusters, but the stick-breaking budget loses its priority semantics.
- **Scope-first, then potency:** group by scope boundary, then sort within scope. Avoids computing scope predicates per-node during the scan. Stick-breaking runs independently within each scope group.

These can be benchmarked empirically. Potency-descending is the recommended starting point.

### 6.3.3 Single-Level Size Budget

For a conversational context with `S` spans averaging `C` centroids each and head dimension `d`:

- Single-level index size: `S × C × (d + 3)` floats ≈ `S × C × (d + 3) × 2` bytes (FP16)
- Example: 500 spans, 4 centroids each, d=128 → 2000 × 131 × 2 ≈ **512 KB**

This fits comfortably in L2 cache on H100 (50 MB). The entire coarse pass can run without touching HBM.

### 6.3.4 Multilevel Size Budget

Each level reduces the number of nodes by the branching factor `b`. The table below assumes 1 centroid per node at each level (the multi-centroid structure from §2.5 is internal to Level 1 spans — each span may have multiple centroids, but the tree node at Level 1 *is* one centroid; a span with 4 centroids produces 4 Level 1 nodes):

| Level | Granularity | Nodes (1T token corpus) | Index Size (d=128, FP16) |
|---|---|---|---|
| 0 | tokens | 10^12 | N/A (raw KV cache, not indexed) |
| 1 | centroids (~25 tok each, assuming ~100-tok spans with ~4 centroids) | ~4 × 10^10 | ~10 TB |
| 2 | regions (b=256 Level 1 nodes each) | ~1.5 × 10^8 | ~40 GB |
| 3 | sectors (b=256 Level 2 nodes each) | ~6 × 10^5 | ~160 MB |
| 4 | domains (b=256 Level 3 nodes each) | ~2,400 | ~600 KB |
| 5 | root (b=256 Level 4 nodes each) | ~10 | ~2.5 KB |

**Residency policy:** Levels 3+ are always GPU-resident (~160 MB combined). Level 2 is GPU-resident for working corpora, host-memory-resident otherwise. Level 1 at corpus scale (~10 TB for 1T tokens) exceeds typical host memory and must itself be paged from storage — only the subset of Level 1 nodes reachable from opened Level 2 nodes needs to be resident during any given query. Level 0 (raw KV cache) is demand-paged from storage.

The total always-resident index for a 1T-token corpus is **~160 MB** (Levels 3–5). This is the "page table" that enables navigating a trillion tokens from GPU memory.

*Note: These numbers are per KV head (§3.5). For a GQA model with h_kv = 8, multiply by 8. They are also approximate and depend heavily on the branching factor and average centroids per span. The key point is the geometric compression: each level is O(1/b) the size of the level below.*

### 6.3.5 Multi-Head Index Strategy

The index stores centroid directions in key space, which is head-specific — different attention heads project keys into different subspaces. Three strategies for handling this:

**Per-KV-head indexes (recommended for GQA/MQA).** Build a separate index per KV head. For GQA with `h_kv` KV heads (typically 4–8), storage cost is `h_kv ×` the single-head figures above. The gating pass runs independently per KV head, and the union of selected spans across heads is compacted for FA-3. With GQA's small `h_kv`, this is practical.

**Shared index over averaged key space.** Compute a single centroid per span by averaging (or concatenating) across KV heads before fitting. Storage cost is 1× regardless of head count, but the index is an approximation — a span relevant to one head but not others may have a diluted centroid. Suitable as a first implementation or when storage is constrained.

**Per-KV-head indexes with shared upper levels.** Build per-head indexes at Level 1, but share the tree structure at upper levels. Upper-level centroids are fit over the concatenation or average of the per-head Level 1 centroids. This gives per-head precision at the leaves (where it matters most for gating accuracy) while keeping upper-level storage shared.

**Recommendation:** per-KV-head indexes for GQA/MQA models (the common case), with the note that the size budget figures in §3.3–3.4 should be multiplied by `h_kv`. For a typical GQA model with `h_kv = 8`, the single-level index grows from ~512 KB to ~4 MB — still comfortably in L2 cache.

At the data structure level, `SpanStore` (§5.6) holds a `Vec<IndexTree>` of length `h_kv` — one tree per KV head. The gating pass runs independently per tree, and the union of selected spans across all KV heads is compacted into a single buffer for FA-3.

## 6.4 Coarse-to-Fine Gating via Stick-Breaking

### 6.4.1 Stick-Breaking Attention: Review

In stick-breaking attention (Tan et al., 2024; ICLR 2025), attention weights are assigned sequentially by consuming a finite budget. For a query at position `j` attending to positions `i` in traversal order:

```
β_i = σ(z_{i,j})                           // break fraction: sigmoid of logit
A_{i,j} = β_i · Π_{k visited after i} (1 - β_k) // attention weight: break × remaining stick
```

The attention weight for position `i` equals the break fraction at `i` times the product of all the stick that survived past subsequent positions. The weights do not sum to 1 in general — the unconsumed remainder `Π_k (1 - β_k)` acts as an implicit null or "attend to nothing" mass. This is a desirable property: softmax forces the model to distribute all weight somewhere, often to low-information sink tokens. Stick-breaking lets the model explicitly decide that the context has nothing useful.

Key properties relevant to span gating:

- **Natural stopping:** when the remaining stick `r = Π (1 - β_k)` falls below `ε`, no further node can receive meaningful weight regardless of its logit. The scan terminates.
- **Implicit ordering without positional encoding:** traversal order determines priority. Combined with NoPE/DroPE, this provides ordering semantics for the coarse pass without requiring position-dependent key encodings.
- **SSM equivalence:** each stick-breaking attention head is equivalent to the end state of a single-gate selective SSM (Gu & Dao, 2023). This opens a path to hardware-efficient implementation via parallel scan.

### 6.4.2 Stick-Breaking over the Span Index

The coarse pass applies stick-breaking attention over the centroid index rather than over individual tokens. The traversal proceeds in potency-descending order:

```
remaining_stick = 1.0
selected_spans = {}

for node_i in index (potency-descending):
    // Compute break fraction from metric logit
    z_i = Metric::logit(query, node_i.centroid)
    β_i = σ(z_i)

    // Allocate from remaining budget
    allocated_i = remaining_stick * β_i
    remaining_stick *= (1 - β_i)

    // Gating decision: open this node for fine attention?
    if allocated_i > τ_open(node_i.coherence):
        selected_spans.insert(node_i.span_id)

    // Early termination
    if remaining_stick < ε:
        break
```

**Key design choices:**

**Break fraction semantics.** The logit `z_i` comes from the metric trait (§2.2) — for `ScaledSphere`, it's `ρ · (q · d)`. The sigmoid maps this to a break fraction in `(0, 1)`. A high-potency span with a direction aligned to the query produces a large logit, consuming a large fraction of the stick. A low-potency or misaligned span takes a small bite.

**Opening threshold `τ_open`.** This threshold has clean semantics: it's the minimum attention budget a node must receive to justify the cost of descending further (materializing a KV cache at the leaf level, or descending into children at upper levels). It can be set based on the compute cost ratio between coarse and fine passes — and in the multilevel case (§4.7), the cost of descending one level deeper vs. accepting the current summary approximation.

**Coherence-adjusted opening.** For low-coherence centroids (where the centroid poorly represents its cluster), the opening threshold is reduced:

```
τ_open(coherence) = τ_base * coherence
```

A coherence of 0.3 means the centroid explains only 30% of its cluster's variance — the actual content might be much more (or less) relevant than the centroid suggests. We err toward opening because skipping a poorly-characterized cluster risks missing highly relevant content that the centroid failed to represent. The cost of a false open (unnecessary fine attention) is wasted compute; the cost of a false skip (missing relevant content) is degraded output quality.

**Per-head independence.** Different attention heads can run the stick-breaking traversal independently over the same index. This is natural — heads specialize in different retrieval patterns, and a span that is irrelevant to one head may be critical to another. The union of selected spans across heads forms the final selection set for compaction. This means some heads will attend (via FA-3) to spans they didn't select — a small waste of compute, but it avoids the complexity of per-head compaction into separate buffers. For GQA/MQA architectures where KV heads are shared, this waste is minimal since fewer independent selection passes are needed.

**Centroid selection opens the full span.** A span with multiple centroids (§2.5) may have only some of its centroids selected by the gating pass. Selecting *any* centroid of a span opens the *entire* span's KV cache for fine attention. The centroid clustering is a summary tool for the gating decision, not a partitioning of the span's tokens — fine attention needs access to all tokens in the span to produce correct results.

### 6.4.3 The Remainder as Null Attention

The stick remainder `r = Π (1 - β_i)` after traversal has a meaningful interpretation: it's the fraction of attention that the model allocates to "none of these spans are relevant." This directly addresses the attention sink problem — rather than forcing weight onto low-information tokens, the model can explicitly abstain.

In soft gating mode (§2.4), the remainder can be mapped to:

- **Zero:** the output is a weighted combination of opened spans only. Simple, but discards the information that the model "looked and found nothing."
- **A learned default vector:** a per-head parameter that represents "background" or "prior" output when no span is relevant. Analogous to the bias term in a linear layer.
- **The current token's own representation:** self-attention as the fallback when context is uninformative.

### 6.4.4 Two-Level Stick-Breaking (Special Case)

The simplest instantiation of the hierarchy uses two levels: stick-breaking over span centroids (Level 1), then fine attention within the selected spans (Level 0). This is the degenerate case of the multilevel tree (§4.7) with a single index level, and is the recommended starting point for implementation.

Two variants at the fine level:

- **Stick-breaking at both levels.** For models trained with stick-breaking attention natively, the fine-grained pass within each opened span also uses stick-breaking. The budget allocated to a span at Level 1 becomes the initial stick for Level 0 within that span. This gives exact budget conservation: total weight across all tokens plus remainder equals 1.0.

- **Stick-breaking gating + softmax fine attention.** Level 1 uses stick-breaking for selection only. Level 0 uses standard softmax (or FA-3) over the selected spans. The stick allocation at Level 1 does not constrain the softmax normalization at Level 0 — it serves purely as a selection mechanism. This works with existing softmax-trained models without retraining.

### 6.4.5 SSM-Equivalent Implementation

The stick-breaking traversal over the centroid index is equivalent to a selective SSM scan:

```
State: (remaining_stick: f32, accumulated_output: Vec<f32>)
Input at step i: (z_i: f32, v_i: Vec<f32>)   // logit and value summary
Gate: β_i = σ(z_i)
Transition:
    accumulated_output += remaining_stick * β_i * v_i
    remaining_stick *= (1 - β_i)
```

This is a 1D selective scan with a scalar gate — the simplest possible SSM. It can be implemented via parallel prefix sum (Blelloch scan) for GPU efficiency, or as a simple sequential loop on CPU when the index is small enough. For the typical index size (~2000 centroids), the sequential version on a single SM is likely faster than the overhead of launching a parallel scan kernel.

### 6.4.6 Scope Interaction

The coarse pass operates **within** the scope set already determined by Mellea's boundary-constrained masking (§1.1). The flow is:

1. **Scope resolution:** determine which spans (Level 1 nodes) are in-scope for this query position, using the span algebra. This is a metadata operation — no tensor computation.
2. **Index filtering:** in the single-level case, select the subset of the centroid index corresponding to in-scope spans. In the multilevel case, scope filtering applies at Level 1 — the tree descent starts at the root and traverses all upper levels normally, but only opens Level 1 nodes that correspond to in-scope spans. (Upper-level nodes do not carry scope metadata; they are structural groupings. A Level 3 node may contain both in-scope and out-of-scope descendants — the scope check happens at the Level 1 leaf.)
3. **Stick-breaking gating:** run the coarse pass (single-level) or tree descent (multilevel) over the filtered set. Produces the selected span set S*.
4. **Fine attention:** attend over S* using stock FA-3 (or stick-breaking at Level 0).

### 6.4.7 Multilevel Recursive Hierarchy

The two-level structure (§4.4) generalizes to an arbitrary-depth tree. The centroid index at Level 1 is itself a set of vectors that can be summarized by higher-level centroids, recursively, forming a tree analogous to the FMM octree:

```
Level L (root):     ~10 nodes          — corpus-level summaries
Level L-1:          ~1K nodes          — domain-level summaries
  ...
Level 2:            ~100K nodes        — region summaries
Level 1:            ~10M nodes         — span centroids
Level 0:            ~1T tokens         — raw KV cache (leaf data)
```

**Tree descent via stick-breaking:**

```
fn descend(query, node, budget, ε, output):
    // Stick-breaking over this node's children
    remaining = budget
    for child in node.children (potency-descending):
        z = Metric::logit(query, child.centroid)
        β = σ(z)
        allocated = remaining * β
        remaining *= (1 - β)

        if allocated > τ_open(child.coherence, child.level):
            if child.is_leaf():
                // Fine attention over child's KV cache.
                // If using stick-breaking at fine level:
                //   fine_attend_sb(query, child.kv_cache, allocated, output)
                //   where `allocated` is the initial stick.
                // If using softmax/FA-3 at fine level:
                //   fine_attend_softmax(query, child.kv_cache, output)
                //   where `allocated` was used only for the gating decision.
                fine_attend(query, child, allocated, output)
            else:
                // Recurse: descend into child with allocated budget
                descend(query, child, allocated, ε, output)
        else if soft_gating_enabled:
            // Far-field approximation: use summary value
            output += allocated * child.value_summary

        if remaining < ε:
            break
```

*Note: this pseudocode is a conceptual illustration of the budget flow. In practice, the descent is split into two phases: (1) collect the selected span set S\* by running the tree descent without performing fine attention, then (2) compact S\*'s KV caches into a contiguous buffer and call FA-3 once over the union. The recursive structure describes the budget allocation logic, not the execution order of attention calls.*

**Key properties:**

**Budget conservation (stick-breaking fine level only).** When stick-breaking is used at all levels, the stick allocated to a parent is partitioned among its children (and the remainder). The total weight across all leaf contributions plus all remainders at every level equals the original stick (1.0). This is exact — no renormalization needed. When softmax is used at the fine level, budget conservation applies only to the index levels; the fine level normalizes independently.

**Adaptive depth.** The tree is not descended uniformly. For a query about "quantum error correction," the root-level stick-breaking pass might allocate 0.7 of the stick to the "physics/computing" domain and 0.01 to "19th-century literature." The physics branch is descended further; the literature branch is either approximated via its summary value (soft gating) or skipped entirely (hard gating). Within the physics branch, "quantum computing" gets most of the 0.7 budget, and within that, "error correction" gets most of what remains. The query reaches the relevant leaf spans in a few stick breaks per level.

**Opening threshold per level.** The cost of opening a node varies by level: opening a Level 4 node means descending into ~256 Level 3 children (cheap — they're all GPU-resident). Opening a Level 1 node means materializing a span's KV cache (potentially expensive — may require re-prefill from storage). The threshold `τ_open` should scale with this cost:

```
τ_open(coherence, level) = τ_base * coherence * cost_factor(level)
```

where `cost_factor(level)` reflects the latency/compute cost of descending from level to level-1. Upper levels are cheap to open (small, resident indexes); lower levels are expensive (demand-paged KV caches).

### 6.4.8 Complexity Analysis

**Coarse pass (index descent):**

Per-query-token work at each level: `O(k × d)` — compute `k` logits (each a dot product in `d` dimensions) and `k` sigmoid/multiply operations for the stick-breaking scan.

Total levels traversed: `O(L) = O(log_b n)`

Coarse pass work per query token: `O(k × d × L)`

**Fine pass (FA-3 over selected spans):**

Work per query token: `O(|S*| × d)` where `|S*|` is the total number of tokens across all selected spans.

**Total work per query token: `O(k × d × L + |S*| × d)`**

The coarse pass is negligible compared to the fine pass. The architectural win comes entirely from `|S*| << n` — the gating pass reduces the token set that FA-3 must process.

**Concrete example (1T tokens, d=128, b=256, k=15):**
- Coarse: `L = ⌈log_256(10^12)⌉ = 5`, so `15 × 128 × 5 = 9,600` ops
- Fine: if gating selects 10 spans averaging 100 tokens → `|S*| = 1,000`, so `1,000 × 128 = 128,000` ops
- Total: ~138,000 ops (dominated by the fine pass)

**Comparison:**

| Method | Complexity per query token | 1T tokens, d=128 |
|---|---|---|
| Full attention | O(n × d) | ~1.3 × 10^14 ops |
| FlashAttention-3 | O(n × d) (same FLOPs, fewer memory ops) | ~1.3 × 10^14 ops |
| Longformer (local + global) | O(w × d + g × d) | Fixed window, misses distant content |
| **Multilevel span attention** | **O(k × d × L + \|S*\| × d)** | **~10^5 ops** (k=15, L=5, \|S*\|=1000) |

The ~10^9× reduction comes from gating: instead of attending to all ~10^12 tokens, FA-3 sees only the ~10^3 tokens in selected spans. The coarse index descent is what makes this selection possible in O(log n) time. Whether the selected set is sufficient — i.e., whether the gating approximation is acceptable — is an empirical question (§10, Phase 8).

## 6.5 Tiered Storage and Eviction

### 6.5.1 The Index as Virtual Memory

The summary index and the KV cache occupy fundamentally different roles:

- **Index:** small, always needed for gating decisions, rarely mutated. This is the **page table**.
- **KV cache:** large, needed only for spans that survive gating, write-once. This is the **physical memory**.

The key insight is that **evicting a span's KV cache need not evict its index entry.** The index entry remains in the tree, participates in stick-breaking traversal, and influences gating decisions. If the gating pass selects an evicted span, that's a **cache fault** — the span's KV cache is re-materialized on demand by re-prefilling from the source tokens.

This decoupling means the index can describe a vastly larger context than what fits in GPU memory (or even host memory) for KV caches. The index is the map; the KV cache is the territory. You can carry the map everywhere and only visit the territory you need.

### 6.5.2 Cache Status Model

Each leaf-level span tracks its KV cache residency:

```
enum CacheStatus {
    GpuResident {
        buffer_offset: usize,         // offset into compacted GPU buffer
        compacted_quant_params: QuantParams, // unified FP8 params post-requantization
    },
    HostResident {
        host_ptr: *const u8,          // pointer into host-memory cache pool
    },
    StorageResident {
        storage_path: PathBuf,        // path to serialized KV cache on disk/object store
    },
    Evicted {
        source_ref: SourceRef,        // reference to source tokens for re-prefill
        estimated_prefill_ms: f32,    // estimated re-prefill latency
    },
}
```

The `quant_params` field in the `LevelIndex` (§3.1) stores each span's **original** quantization parameters as computed at index time. The `compacted_quant_params` in `GpuResident` stores the **unified** parameters assigned during the requantization pass (§5.5), which may differ from the originals. Both are needed: the originals are used if a span is re-loaded from storage (where it's stored with original quantization); the unified params are what FA-3 sees in the compacted buffer.

The gating pass always runs against the full index regardless of cache status. The cache status only matters after gating, when selected spans need their KV caches materialized.

### 6.5.3 Tiered Residency

The storage tiers form a hierarchy mirroring the index tree:

| Tier | Medium | Capacity (typical) | Latency | Contents |
|---|---|---|---|---|
| T0 | GPU HBM | 80 GB | ~1 μs | Active working set KV caches + all upper-level indexes |
| T1 | Host RAM | 512 GB – 2 TB | ~10 μs | Warm KV caches + Level 1–2 indexes for large corpora |
| T2 | NVMe SSD | 4–32 TB | ~100 μs | Cold KV caches + serialized indexes |
| T3 | Object store | Unbounded | ~10 ms | Archival KV caches + corpus-level indexes |
| Evicted | Source tokens | Unbounded | ~1–100 ms (prefill) | No KV cache; re-prefill on demand |

**Promotion and demotion** happen during the async maintenance window (§5.6):

- **Promote:** when a span's stick allocation trends upward across recent turns, speculatively promote its KV cache to a faster tier. For evicted spans, speculatively re-prefill.
- **Demote:** when a span hasn't been selected in the last N turns, demote to a slower tier. Eventually evict (delete KV cache, retain index entry).

### 6.5.4 Cache Fault Handling

When the gating pass selects a span whose KV cache is not GPU-resident:

1. **Host-resident (T1):** async DMA to GPU. Latency ~10–100 μs depending on span size. Can be overlapped with fine attention over other selected spans that are GPU-resident.
2. **Storage-resident (T2/T3):** async read + DMA. Latency ~100 μs – 10 ms. If the query is latency-sensitive, use the span's summary value as an approximation (soft gating fallback) while the full cache loads asynchronously.
3. **Evicted:** re-prefill from source tokens. This is the most expensive fault — it requires a forward pass through the model for the span's source tokens to recompute KV projections. Latency depends on span length and model size. Mitigation: track cache fault rate and proactively re-prefill spans whose stick allocation is trending upward.

**Graceful degradation:** if a cache fault would exceed a latency budget, the system can fall back to the span's summary value (soft gating) or skip the span entirely (hard gating). Quality degrades gracefully — the system never blocks indefinitely. The output is always valid, just potentially less precise.

### 6.5.5 Operations

Four maintenance operations run asynchronously between user messages:

| Operation | Trigger | Description |
|---|---|---|
| **Reindex** | New span added to store | Fit centroids for the new span; insert into index tree at Level 1; aggregate potency and refit centroids upward to root |
| **Defragment** | Selection set changed | Compact the KV cache of likely-needed spans into contiguous GPU memory |
| **Requantize** | Defragmentation or new span | Standardize FP8 scaling factors across compacted spans for FA-3 |
| **Tier manage** | Per-span access statistics | Promote/demote KV caches across storage tiers based on stick allocation trends; speculatively re-prefill evicted high-trend spans |

### 6.5.6 Consistency Model: Double-Buffered Index

To avoid blocking inference on index updates, the index is double-buffered:

```
SpanStore {
    active_indexes: Arc<Vec<IndexTree>>,     // one per KV head; immutable, used by inference
    building_indexes: Mutex<Vec<IndexTree>>, // one per KV head; mutable, updated by maintenance
    generation: AtomicU64,
}
```

Swap protocol:

1. Maintenance thread acquires `building_indexes`, applies updates across all KV heads.
2. When complete, atomically swaps `active_indexes` and `building_indexes` (Arc swap).
3. Inference always reads from `active_indexes` without locking.

This is the RCU (read-copy-update) pattern. Rust's ownership model enforces that the active index is never mutated while readers hold references.

**Staleness guarantee:** The active index is at most one generation behind. For conversational workloads (seconds between messages), the maintenance window is more than sufficient to complete a full reindex/defragment/requantize/tier-manage cycle.

### 6.5.7 Incremental Reindexing

Full reindex (refit all centroids at all levels, re-sort) is `O(Σ_l N_l × d)` where `N_l` is the number of nodes at level `l`. For large trees, this can be done incrementally:

- **New span:** fit Level 1 centroids, insert into Level 1 index. Propagate upward: update the parent node's potency (sum of children) and refit its centroid direction by incorporating the new children (incremental mean update). Continue propagation to root. `O(L × d)` total.
- **Evicted span (KV cache only):** no index change. Set `cache_status = Evicted`. `O(1)`.
- **Deleted span (index + cache):** remove centroid entries at Level 1, propagate potency decrements and centroid refits upward. `O(L × d)`.

Full re-sort and tree rebalancing are deferred to periodic compaction during the maintenance window.

### 6.5.8 Defragmentation and Pre-Compaction

Based on recent gating decisions (which spans were selected in the last N turns), maintain a **predicted working set** of spans likely to be selected next. During the maintenance window:

1. Identify the predicted working set (e.g., spans selected in ≥ 1 of last 3 turns, plus high-potency newcomers, plus spans with rising stick allocation trends).
2. Ensure their KV caches are GPU-resident (promote from lower tiers if needed).
3. Copy their KV cache blocks into a contiguous buffer, ordered by span sequence.
4. Compute unified FP8 quantization parameters across the compacted buffer.
5. Store the compacted buffer alongside the active index; fine-grained attention reads from it directly.

If the actual gating decision at inference time selects a span not in the pre-compacted set, fall back to gathering from the main span store (slower, but correct). Track hit rate to tune the working set predictor.

## 6.6 Offline Precomputation

### 6.6.1 Corpus Indexing Pipeline

The multilevel index for a large corpus can be built entirely offline:

```
Pipeline: corpus → tokenize → encode → build Level 1 → ... → build Level L

Stage 1: Tokenize and segment corpus into spans (paragraph breaks,
         section boundaries, or fixed-length chunks)
Stage 2: Run model forward pass to compute KV projections for each span.
         In a hybrid SSM-attention model, only the attention layers produce
         KV caches; SSM layer states are transient. A model with 4 attention
         layers and 28 SSM layers stores KV for only 4 layers, significantly
         reducing per-span cache size.
Stage 3: Fit Level 1 centroids over each span's key vectors
Stage 4: Cluster Level 1 centroids into Level 2 regions; aggregate potency
Stage 5: Repeat upward to root
Stage 6: Serialize: upper-level index tree (Levels 2+ always shipped) +
         Level 1 index (partitioned for selective loading) +
         KV caches (shipped or regenerated on demand)
```

**Parallelism:** Stages 1–3 are embarrassingly parallel across spans. Stages 4–5 are hierarchical bottom-up aggregations, parallelizable within each level.

**Position invariance is essential.** Because the attention layers use position-invariant key projections (via NoPE or DroPE within a hybrid architecture, §1.2), the KV caches and centroids computed in Stage 2–3 remain valid regardless of how the spans are later composed or which subset is selected at inference time. A corpus indexed once can be queried by any model instance without recomputation.

### 6.6.2 Index Distribution

The index tree is small enough to distribute independently of the KV caches:

- A 1T-token corpus has a ~160 MB always-resident index (Levels 3–5).
- The Level 1–2 indexes add ~10+ TB — too large to ship wholesale, but partitionable by domain/region for selective loading.
- The KV caches themselves are tens of TB — stored on object storage, demand-paged.

This means a deployment can ship the upper-level index with the model and treat the lower-level indexes and KV caches as remote, lazily-loaded data stores. The upper index tells the model what exists and roughly where it is; the lower levels and KV caches provide precision when needed.

### 6.6.3 Incremental Corpus Updates

When new documents are added to a pre-indexed corpus:

1. Compute KV caches and Level 1 centroids for the new spans (offline).
2. Insert into the index tree (incremental reindex, §5.7).
3. Propagate summary updates upward to root.
4. Ship the delta (new index entries + new KV caches) to serving infrastructure.

Deletions are handled by tombstoning at Level 1 and propagating reduced potency upward. The tree remains valid (deleted spans simply receive no stick allocation because their centroids are removed from traversal); periodic compaction reclaims space.

## 6.7 FA-3 Integration

The hierarchical gating resolves the FA-3 constraints described in §1.3:

| FA-3 Constraint | Resolution |
|---|---|
| Contiguous memory layout | Pre-compacted by defragmentation (§5.8) |
| Uniform FP8 quantization | Standardized during requantization (§5.5) |
| Regular mask topology | Coarse gating reduces the active span set; remaining mask is typically dense (all selected spans in-scope) or a simple block mask determined by scope rules |
| Global softmax normalization | Stick-breaking at upper levels handles variable scope; FA-3 at leaf level sees a fixed, pre-selected span set |

The fine-attention call to FA-3 receives:

- Contiguous Q from the current generation
- Contiguous pre-compacted KV from the selected spans
- A mask descriptor (typically dense; or a block mask reflecting scope boundaries when partial visibility applies)
- Uniform FP8 quantization parameters

No kernel modifications required. The multilevel stick-breaking descent is a separate, lightweight operation that completes before FA-3 is invoked.

## 6.8 Related Work

**Stick-breaking attention.** Tan, Kim, Liu et al. (2024; ICLR 2025) introduced stick-breaking as a drop-in replacement for softmax attention, demonstrating competitive performance with improved length generalization. Their key insight — that each stick-breaking head is equivalent to a selective SSM — motivates our use of stick-breaking for tree descent, where the SSM structure enables efficient sequential traversal with natural early termination at every tree level. The connection to NoPE (no positional embeddings) is particularly relevant: stick-breaking provides implicit ordering via traversal sequence, which pairs naturally with Mellea's position-invariant span algebra.

**Barnes-Hut / Fast Multipole Method (FMM).** The N-body simulation literature provides the structural template. Barnes-Hut (1986) introduced hierarchical spatial clustering with an opening-angle criterion: a distant cluster of particles is approximated by its center-of-mass unless the ratio of cluster size to distance exceeds a threshold θ, in which case the cluster is "opened" and its children are examined individually. This yields O(n log n) force computation. The FMM (Greengard & Rokhlin, 1987) extended this to O(n) via multipole expansions at every level — far-field interactions are computed entirely from summaries, without ever examining individual particles. Our architecture is closer to FMM than to Barnes-Hut: the summary values at each tree level serve as multipole-like far-field approximations, and the stick-breaking budget serves as the opening-angle criterion. The key adaptation is that "distance" in our setting is angular distance in key space rather than Euclidean distance in physical space, and "mass" is potency (aggregate key norm).

**Hierarchical attention.** Several works have explored multi-resolution attention (e.g., Longformer, BigBird, hierarchical transformers). These typically fix the hierarchy at the token level (local + global attention patterns). Our approach operates at the span level, where the hierarchy is semantic (span boundaries carry meaning) rather than positional (every k-th token), and extends to an arbitrary-depth tree for corpus-scale contexts.

**Attention sinks.** Xiao et al. (2023) identified that softmax attention concentrates weight on low-information tokens to effectively "do nothing." The stick-breaking remainder (§4.3) provides a principled mechanism for this: the model can explicitly allocate attention to "nothing" by consuming less of the stick, rather than being forced to park weight on sink tokens. In the multilevel tree, a remainder at any level means "nothing in this subtree is relevant" — the unallocated budget is simply not passed to child nodes, because the budget was never allocated to the parent in the first place.

**Retrieval-augmented generation (RAG).** Standard RAG pipelines retrieve documents via a separate retrieval model (e.g., dense passage retrieval) and then attend over the retrieved passages. Our architecture unifies retrieval and attention: the multilevel stick-breaking descent *is* the retrieval operation, performed in the model's native key space. This eliminates the representation mismatch between retrieval and generation models and allows end-to-end training of the retrieval hierarchy.

**Position-invariant attention in hybrid architectures.** The entire architecture depends on position invariance of KV caches. Pure NoPE (Kazemnejad et al., 2024) demonstrated that transformers without positional embeddings can implicitly recover positional information, but no scaled NoPE-only models exist in practice. The practical path is hybrid SSM-attention architectures — Granite 4-h (IBM), Jamba (AI21), Bamba (IBM Research), Samba (Ren et al., 2024; ICLR 2025), Nemotron-H (NVIDIA) — where SSM layers provide implicit sequential structure and attention layers provide precise retrieval. In these hybrids, the attention layers can use NoPE (as in Samba-NoPE) since the SSM layers handle positional awareness. DroPE (Gelberg et al., 2026; Sakana AI) provides a retrofit path: drop positional embeddings from attention layers of an existing RoPE-trained model and recalibrate at ~0.5% of pretraining cost. This is particularly relevant because it enables applying this architecture to existing model families (including IBM's Granite family) without full retraining.

## 6.9 Open Questions

1. **Centroid fitting algorithm.** Spherical k-means vs. vMF EM vs. simpler heuristics (PCA of key matrix → top components as centroids). Trade-off between fit quality and compute cost at store time. For offline precomputation, compute cost is less critical — quality matters more.

2. **Optimal expansion order heuristic.** BIC/AIC on the vMF mixture, or a simpler proxy like effective rank of the key matrix? Needs empirical validation.

3. **Training-time integration.** For models trained with stick-breaking attention natively, the multilevel hierarchy (§4.7) is a natural extension. For softmax-trained models used at inference only, the coarse stick-breaking pass is a learned component that needs its own training signal. Options include: (a) distillation from full attention outputs, (b) RL with approximation error as reward, (c) connection to modern Hopfield networks — Dima Krotov's work on dense associative memory may offer a framework where the coarse/fine distinction emerges from inverse temperature β. Discuss with Dima.

4. **Cross-conversation shared index.** Position-invariant spans could in principle be shared across conversations with a single global index. The multilevel tree makes this more natural — the upper levels describe the corpus, not the conversation. Raises consistency, privacy, and eviction policy questions.

5. **Interaction with speculative decoding.** The coarse pass could inform speculative decode candidates — if a span is highly activated in the summary, its content is likely to influence the next token. Worth exploring.

6. **Metric variant selection.** The `ScaledSphere` factorization (direction × scalar potency) assumes that norm and direction are approximately independent. When they're not (e.g., architecturally-induced correlations), the approximation degrades. Need empirical measurement of factorization error across model families.

7. **Traversal order learning.** Potency-descending is a reasonable heuristic, but the optimal traversal order is query-dependent. A learned reordering (e.g., a small network that predicts traversal priority from the query) could improve budget allocation at the cost of an additional forward pass. Trade-off analysis needed.

8. **Stick-breaking temperature.** The sigmoid in the break fraction `β = σ(z)` can be tempered: `β = σ(z / T)`. Low temperature → sharper gating (more spans skipped); high temperature → softer gating (more spans opened). This is a tunable quality-vs-speed knob at inference time, analogous to beam width in search. Can vary by tree level — tighter at upper levels (cheap to be conservative) and looser at lower levels (expensive to open unnecessarily). Connection to Hopfield inverse temperature β is worth formalizing.

9. **Tree balancing and fan-out.** The optimal branching factor may vary by level. Upper levels covering heterogeneous content benefit from higher fan-out (more centroids per node). Lower levels covering coherent spans may need less. Adaptive fan-out adds complexity but could improve gating quality. The relationship between fan-out and stick-breaking early termination needs empirical characterization.

10. **Approximation error bounds.** Can we bound the approximation error (L2 distance between hierarchical-gated output and full-attention output) as a function of tree depth, branching factor, and stick temperature? An analytic bound would inform the quality-vs-speed trade-off and guide threshold selection. The FMM literature has well-characterized error bounds via multipole expansion order — the analog here would be the number of centroids per node and the coherence of each cluster.

11. **Multi-model index compatibility.** If two models share the same key-space geometry (e.g., same architecture, different training runs), can they share a pre-built index? The centroids are fit in key space, which is model-specific. However, if models are fine-tuned from a common base, the key spaces may be sufficiently aligned for index sharing, avoiding redundant offline computation.

12. **Scope-potency interaction in the multilevel tree.** A parent node's potency is the sum of all its descendants' potencies, including Level 1 nodes that may be out-of-scope for a given query. This inflates the parent's potency, causing it to consume more stick budget than the in-scope portion warrants. The wasted budget is recovered at Level 1 (out-of-scope nodes are filtered), but the inflation can starve other branches of budget they deserved. Possible mitigations: scope-aware potency (requires per-query recomputation — expensive), potency discounting based on the fraction of in-scope descendants (requires scope metadata at upper levels), or simply accepting the waste as a cost of scope-agnostic indexing. Empirical measurement of the budget distortion is needed.

## 6.10 Implementation Plan

### Phase 1: Span Summary Computation
- Implement `SpanSummary<ScaledSphere>` in `mellea-core` (Rust)
- Centroid fitting via spherical k-means (simplest viable algorithm)
- Separate potency aggregation (sum of children) from centroid direction fitting
- Unit tests against known key distributions

### Phase 2: Single-Level Index Structure
- `LevelIndex` with double-buffer swap
- `CacheStatus` enum and tiered residency metadata
- PyO3 bindings for Python integration / prototyping
- Benchmark index construction and query latency

### Phase 3: Stick-Breaking Gating Kernel
- Sequential stick-breaking scan over centroid index
- Early termination on remaining stick < ε
- Coherence-adjusted opening threshold
- Per-head independent traversal with union selection
- Integration with Mellea scope resolution

### Phase 4: Single-Level End-to-End Pipeline
- Wire coarse (stick-breaking) → gate → fine (FA-3) pipeline
- Pre-compaction and defragmentation in async worker
- FP8 requantization pass
- Benchmark against baseline (full FA-3 over all in-scope spans)

### Phase 5: Tiered Eviction
- Implement `CacheStatus` transitions and tier management
- Cache fault handling (host-resident → GPU DMA, evicted → re-prefill)
- Speculative promotion based on stick allocation trends
- Soft gating fallback for high-latency cache faults

### Phase 6: Multilevel Index Tree
- `IndexTree` structure with recursive `LevelIndex` construction
- Bottom-up index building from Level 1 centroids
- Top-down stick-breaking descent with per-level thresholds
- Incremental insertion and upward propagation

### Phase 7: Offline Precomputation Pipeline
- Corpus segmentation → KV projection computation → hierarchical index building
- Serialization format for index tree + KV cache references
- Delta update protocol for incremental corpus additions

### Phase 8: Evaluation
- Approximation error: compare hierarchical-gated output vs. full attention output, measured at each tree depth
- Latency: measure speedup as a function of corpus size and gating aggressiveness (stick temperature per level)
- Cache fault rate and impact on tail latency
- Working set prediction hit rate
- Ablation: stick-breaking vs. softmax-based threshold (v0.1 design) for gating quality
- Scale test: demonstrate O(k log n) scaling from 10^6 to 10^12 tokens

---

> [← Ch.5: Serving Engine](05-serving-engine.md) · [Table of Contents](../README.md) · [Ch.7: Mellea Standard Library →](07-mellea-stdlib.md)
