> [← Ch.3: Model-Runtime Codesign](03-model-runtime-codesign.md) · [Table of Contents](../README.md) · [Ch.5: Serving Engine →](05-serving-engine.md)

---

# Chapter 4: Cache-Materialized Training

**Component:** `mellea-core`
**Status:** Draft
**Version:** 0.1
**Authors:** David Cox
**Date Created:** 2026-04-01
**Date Modified:** 2026-04-06
**Scope:** Single-node implementation on LM Engine (8× H100 or GB200 NVL72). Dense transformer and Granite hybrid SSM-attention architectures.

**Prerequisites:** Chapter 1 (Span Algebra — free monoid, scope composition), Chapter 2 (LLMON §2.6–2.8 — span boundary extraction, dependency DAGs)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---


## 4.1 Problem

Standard transformer pretraining treats the KV cache as a pure inference artifact. During training, the model sees a full causal attention matrix over the entire sequence. During inference, it reads and writes a KV cache incrementally. These are different computational regimes.

This asymmetry becomes a problem when we want the model to reason about cache structure. Span composition, cache compression, learned eviction, differentiable retrieval—these are all capabilities that would benefit from the model producing cache entries that are useful when composed, compressed, or selectively retained. But none of them can be properly explored in a training regime where the cache doesn't exist. Training with full attention and hoping cache-aware behaviors emerge from block-mask proxies is indirect at best.

Cache-materialized training closes this gap. The model processes sequences as ordered spans, explicitly materializing KV caches at span boundaries. Cross-span attention reads from the materialized cache. The cache write path is in the computation graph, so gradients flow through it. The model learns in the regime it will inhabit at serving time.

The core proposal is this: materialize the cache during training. Write K,V at span boundaries, attend to them in subsequent spans, let gradients flow through the write. No new parameters, no new modules—just restructuring the forward pass to match inference. Once this machinery is in place, it *enables* a family of optional extensions—learned compression, eviction, merging—that become natural additions to the cache write path. But the extensions are not the proposal. The proposal is the materialization itself.

## 4.2 Key Insight: DRoPE/NoPE SimplifiesExecution

Under standard RoPE, each key vector encodes its absolute position. Composing KV caches from different spans requires re-encoding or re-indexing positions, introducing complexity and a potential train/inference mismatch. Under DRoPE (decoupled rotary position embedding) or NoPE (no positional encoding in attention), position information is either absent from the key vectors or carried as a separable signal. Cache entries from different spans compose by concatenation with no re-indexing.

This is the same property that makes span composition a free monoid in the Mellea span algebra. It also makes cache materialization during training clean: you write K,V into the cache without worrying about position entanglement. The algebraic simplification and the training simplification have the same root cause.

## 4.3 Two-Phase Curriculum

### Phase 1: Standard Pretraining

No changes. Full causal attention, teacher forcing, maximum parallelism. The model learns language and base attention patterns. This phase consumes ~90–95% of total training tokens and runs on existing infrastructure at full speed.

### Phase 2: Cache-Materialized Training

At a curriculum boundary, training transitions to a span-aware regime. Each training sequence carries semantic span annotations—document boundaries, conversational turns, instruction/response pairs, retrieved passages, or other structurally meaningful units determined by the mid-training data mixture. Within each span, computation is fully parallel (standard causal attention). Between spans, KV caches are materialized and composed via identity writes—raw K,V concatenation, no new parameters. Cross-span attention reads from the accumulated cache. Gradients flow backward through the cache write path, teaching the model's existing K,V projections to produce cache entries that are useful when composed.

Phase 2 consumes ~5–10% of total training tokens. For a model trained on 2T tokens total, Phase 2 is 100–200B tokens. The model already knows language; Phase 2 teaches it to be aware of its own inference-time data structures. Optional learned transforms (compression, eviction, merging) can be explored after the core materialization is validated.

The transition is analogous to established mid-training curriculum shifts: context extension, learning rate annealing, capability-specific fine-tuning. The training dynamics (loss spike at transition, recovery over a few thousand steps) are expected and well-understood.

## 4.4 Spans Are Determined by Data Structure

The span boundaries in cache-materialized training are not arbitrary chunking. They are the semantically meaningful units that the Mellea span algebra operates over: documents within a multi-document context, turns in a conversation, system prompt vs. user prompt vs. assistant response, retrieved passages in a RAG setting, tool calls and their results.

This is what makes cache materialization meaningful rather than merely structural. The model learns to produce cache entries that represent a *document*, a *turn*, a *retrieved passage*—not an arbitrary 512-token slice. When optional compression or eviction is later applied, it operates on these coherent semantic units rather than arbitrary token windows.

Mid-training data is curated with this structure in mind. Training sequences are composed of structured objects—documents, conversational turns, retrieval results, tool invocations—serialized into the token stream via LLMON, a token-native structured object serialization format that uses special tokens to delimit object boundaries. The span boundaries are inherent in the structure of these objects, not annotations added after the fact. The data pipeline extracts span boundaries from the LLMON structure at preprocessing time and provides them alongside the token sequence. Span lengths vary according to the natural length of the underlying semantic unit—a short user turn might be 20 tokens, a retrieved document might be 2000 tokens, a system prompt might be 500 tokens.

This has implications for the feasibility analysis: span counts and span lengths are properties of the data distribution, not tunable hyperparameters. The analysis below uses representative values from typical mid-training mixtures.

## 4.5 Algorithm

### 4.5.1 Span-Sequential Forward Pass

Given a sequence of S tokens with span boundaries at positions [b₀, b₁, ..., b_N] determined by data structure, process N spans in order:

```
initialize cache = {} (empty, per-layer)

for j in 1..N:
    span_j = tokens[b_{j-1} : b_j]     # variable-length, semantically determined
    
    for each layer:
        Q, K, V = project(span_j)
        
        # within-span: standard causal attention
        attn_within = causal_attention(Q, K, V)
        
        # cross-span: attend to accumulated cache
        if cache is not empty:
            K_cache, V_cache = cache[layer]
            attn_cross = cross_attention(Q, K_cache, V_cache)
            attn = combine(attn_within, attn_cross)
        else:
            attn = attn_within
        
        # cache write (identity in core; optionally transformed)
        K_new, V_new = K, V     # identity write — the core proposal
        cache[layer] = concatenate(cache[layer], K_new, V_new)
        
        span_j = feedforward(attn)
    
    loss += cross_entropy(span_j, labels_j)

loss.backward()  # gradients flow through cache writes
```

In practice, the within-span and cross-span attention are fused into a single FlashAttention call: the query block is the current span's Q, and the KV block is the concatenation of the current span's K,V with the accumulated cache. FlashAttention's variable-length (`varlen`) interface handles variable-length spans natively—this is the same interface LM Engine already uses for padding-free transformers.

### 4.5.2 Cache Writes

At each span boundary, the K and V projections for that span are written into the cache. In the core algorithm, this is an identity operation—raw K,V tensors are concatenated into the accumulated cache. No new parameters, no new modules. The model's existing K,V projections receive gradients through the cache path, teaching them to produce cache entries that are useful when composed across spans.

This is the full core proposal. Everything below in Section 5.3 is an optional extension enabled by the machinery.

### 4.5.3 Optional Extension: Learned Cache Transforms

Because the cache write path is in the computation graph, it becomes a natural site for inserting differentiable transforms. These are **not required** for cache-materialized training to be useful—the identity write already closes the train/serve gap and enables span composition. But once the machinery exists, parameterized transforms at the write path become straightforward additions that unlock cache-aware behaviors the model cannot learn with identity writes alone.

The `cache_transform` at each span boundary is a differentiable function F(K, V; θ) → (K', V'). Possible instantiations include:

**Compression.** A learned projection reduces a span's cache entries to a smaller representation. The model learns what information to preserve for downstream attention, supervised end-to-end by the training loss. No auxiliary objective or distillation target—just backprop through the cache. Because spans are semantically meaningful, compression operates on coherent units: "compress this document's cache" rather than "compress these 512 arbitrary tokens."

**Eviction.** A learned scorer assigns importance to each cache entry. Low-scoring entries are masked out via Gumbel-softmax (differentiable approximation to top-k). The model learns content-dependent retention policies rather than relying on hand-designed heuristics (LRU, sliding window, attention-score thresholding).

**Merging.** Instead of growing the cache by concatenation (free monoid), a learned merge operator folds new entries into a fixed-size cache via cross-attention. This bounds cache size regardless of the number of composed spans, moving from a free monoid to a learned monoid with a parameterized composition operator.

These transforms are not mutually exclusive. A span boundary can apply a pipeline: evict low-value entries, compress survivors, merge into fixed-size cache. Each stage is differentiable, and gradients compose through the pipeline.

Each of these extensions introduces new parameters that must be initialized and trained during Phase 2. They are best introduced incrementally, after the core cache materialization is validated (see the curriculum in Section 9).

### 4.5.4 Gradient Flow

The critical property: gradients from the training loss propagate backward through cross-span attention and through the cache write.

With identity writes (the core proposal), no new parameters are trained. But the model's K,V projection matrices still receive gradients through the cache path. This is the key training signal that standard full-attention training lacks: the K,V projections learn that their outputs will be *cached and reused across span boundaries*, not just consumed within the current attention window.

If optional learned transforms are added to the write path, their parameters receive a direct training signal: "did transforming these cache entries help or hurt the model's ability to use them in later spans?" No auxiliary loss, no distillation target, no reinforcement signal. Just backprop.

### 4.5.5 Gradient Checkpointing at Span Boundaries

Retaining the full computation graph across all N spans is memory-intensive. Span boundaries provide a natural checkpointing structure: retain the materialized cache (small: just K,V tensors) and discard within-span activations. During the backward pass, recompute within-span activations from the span's input tokens and the cached KV state at that boundary.

Cost: one additional within-span forward pass per span during backward. This is the standard activation checkpointing tradeoff (memory for compute), applied at the span granularity. It composes with layer-level checkpointing within each span.

## 4.6 Span Parallelism

### 4.6.1 The Dependency Structure Is a DAG, Not a Chain

The algorithm in Section 5 imposes a total order on spans: span 1 → span 2 → span 3 → ... → span N. But the actual dependency structure in most mid-training data is a partial order. Consider a RAG-style sequence:

```
[system prompt] → [retrieved doc A]
                   [retrieved doc B]  → [user turn] → [assistant response]
                   [retrieved doc C]
```

Docs A, B, and C each depend on the system prompt's cache (they attend to it), but they don't depend on each other. The user turn depends on all of them. The dependency graph is a DAG with width—independent spans that can be processed concurrently.

The same pattern recurs across mid-training data formats: multi-document contexts (independent documents sharing a common preamble), multi-turn conversations with parallel tool calls (the model issues three tool calls simultaneously; the results are independent), and structured objects with sibling children in the LLMON hierarchy.

The LLMON serialization format necessarily linearizes this structure into a token sequence, but the underlying object graph preserves the DAG. The data pipeline can extract this dependency structure alongside the span boundaries and provide it to the training loop.

### 4.6.2 Why the Algebra Licenses This

Span parallelism is correct because of the free monoid property. Composition is concatenation; concatenation is associative:

```
compose(system, compose(docA, compose(docB, docC)))
= compose(system, docA, docB, docC)
= compose(system, compose(docC, compose(docA, docB)))
```

The order of composition among independent spans doesn't affect the result. A query in the user turn attending to the composed cache [system ‖ docA ‖ docB ‖ docC] produces the same attention output regardless of the order in which the docs were concatenated.

This is the sense in which the algebraic framework earns its keep. The free monoid property isn't just an elegant abstraction—it's what makes span parallelism *correct by construction*. No correctness proof is needed beyond the algebraic guarantee.

### 4.6.3 Processing Model

Given the dependency DAG, process spans in waves. Each wave contains all spans whose dependencies have been satisfied by previous waves:

```
initialize cache = {}
dag = batch["span_dag"]     # from data pipeline

for wave in topological_waves(dag):
    # All spans in this wave are independent of each other
    # but depend on the cache from prior waves
    
    # Batch all spans in the wave into a single kernel call
    all_q, all_k, all_v = [], [], []
    for span in wave:
        q, k, v = project(span.tokens)
        all_q.append(q)
        all_k.append(k)
        all_v.append(v)
    
    # Fused FlashAttention: batch of variable-length spans,
    # each attending to its own tokens + the shared cache
    attn_out = flash_attention_varlen_batch(
        all_q, all_k, all_v, kv_cache=cache
    )
    
    # Cache write: all spans in the wave write simultaneously
    for span, k, v in zip(wave, all_k, all_v):
        cache = compose(cache, (k, v))  # identity write; concatenation
    
    # Loss for all spans in this wave
    loss += compute_loss(attn_out, wave)

loss.backward()
```

The topological wave decomposition of the DAG determines the sequential depth. For the RAG example above:

- Wave 0: [system prompt] — 1 span
- Wave 1: [doc A, doc B, doc C] — 3 spans in parallel
- Wave 2: [user turn] — 1 span
- Wave 3: [assistant response] — 1 span

Sequential depth = 4 waves instead of 6 serial spans.

### 4.6.4 GPU Utilization Impact

Span parallelism and chain fusion together eliminate the kernel launch overhead problem almost entirely.

**Chain fusion.** When consecutive spans form a chain (each depends on all prior), the attention is mathematically identical to standard causal attention over the concatenated tokens. A single fused FlashAttention kernel processes the entire chain; span boundaries are applied *after* attention by slicing the K,V outputs for cache materialization. This is a special case of kernel fusion admitted by the span algebra: the causal structure of the chain matches the standard causal mask.

**Span-parallel batching.** Independent spans within a wave are batched into a single `varlen` FlashAttention call, so many short spans saturate the GPU as a single aggregate call.

The two compose: chain segments within the DAG are fused into single kernels; independent spans across DAG branches are batched into single kernels. Separate kernel launches only occur at wave boundaries—where the cache context genuinely changes.

For the RAG example: the system prompt is one kernel, the three independent docs are one batched kernel, the user turn is one kernel, the response is one kernel. Four kernel launches total instead of the naive six.

With chain fusion and span-parallel batching, the residual overhead is proportional to the number of *wave boundaries* in the DAG—not the total number of spans. For data with high DAG width or long chain segments, the overhead approaches zero:

| Scenario | Total spans | Sequential depth | Overhead |
|---|---|---|---|
| Fully sequential (chain) | 16 | 1 (fused) | ~0% (single kernel) |
| RAG with 5 docs | 8 | 4 waves | ~1.05–1.10× |
| Multi-tool-call with 4 tools | 10 | 5 waves | ~1.08–1.12× |
| Multi-document context, 8 docs | 10 | 3 waves | ~1.03–1.06× |

### 4.6.5 Gradient Flow in the DAG

In the sequential regime, gradients from the user turn flow back through doc C's cache, then B's, then A's—an artificial chain dependency. In the parallel regime, gradients from the user turn flow back through all three docs' caches simultaneously, but there is no gradient path *between* independent docs. Doc A's cache representation is not influenced by what doc B said.

This is arguably more correct. The gradient signal for doc A's cache write is: "how useful was doc A's cache for the downstream user turn?" This is the right question. The sequential regime asks a subtly different question: "how useful was doc A's cache, given that doc B and C's caches were already in the accumulated cache when doc A was processed?" The parallel regime isolates the signal per span.

For the optional merge operator (Section 5.3), span parallelism raises an additional question: if independent spans merge into a shared cache simultaneously, the merge must be commutative (order-independent) in addition to associative. Concatenation is both. A learned merge may not be. If span-type-aware merging is used (open question 7 in Section 12), commutativity of the merge operator for independent spans is a design constraint that should be enforced architecturally.

### 4.6.6 Data Format Extension

The data pipeline provides the span dependency DAG alongside boundaries and types:

```yaml
# Extended data format
#   input_ids: [token_ids...]
#   labels: [label_ids...]
#   span_boundaries: [[0, 487], [487, 1203], [1203, 2891], [2891, 3402], [3402, 8192]]
#   span_types: ["system", "document", "document", "turn", "response"]
#   span_deps: [[], [0], [0], [0, 1, 2], [0, 1, 2, 3]]
#                ^     ^     ^     ^            ^
#                |     |     |     depends on   depends on
#                root  depends on  system +     everything
#                      system      all docs
```

`span_deps[i]` lists the indices of spans that span i depends on (i.e., whose caches span i needs to attend to). The topological wave decomposition is computed from this at training time—a negligible cost.

For sequences with fully sequential dependencies (e.g., a simple multi-turn conversation where each turn depends on all prior turns), `span_deps` encodes a chain and the algorithm degenerates to the sequential version in Section 5. No special-casing is needed.

### 4.6.7 When Parallelism Is Unnecessary

Not all data has DAG width. A multi-turn conversation where each turn depends on all prior turns is a chain:

```
turn_1 → turn_2 → turn_3 → turn_4 → turn_5
```

Here every wave has width 1, and span parallelism provides no benefit. But this is actually the *best* case for overhead, not the worst: chain fusion means the entire chain runs as a single fused FlashAttention kernel with zero launch overhead. Span parallelism is the optimization for DAG-shaped data; chain fusion is the optimization for chain-shaped data. Between them, there is no bad case.

The mid-training data mixture will contain both patterns. The training loop processes whatever DAG the data provides—the code path is the same, and both chain and DAG structures are handled efficiently.

## 4.7 Feasibility

### 4.7.1 Reference Configuration

| Parameter | Value |
|---|---|
| Model | 8B parameters (Granite-scale) |
| d_model / heads / d_head | 4096 / 32 / 128 |
| Layers | 32 |
| Sequence length S | 8192 |
| Typical spans per sequence N | 4–20 (data-dependent) |
| Typical span length | 200–2000 tokens (data-dependent) |

For the numerical estimates below, we use N=16 spans with an average span length of 512 tokens as a representative working point. Actual values depend on the mid-training data mixture—a multi-document pretraining sequence might have 4 long spans, while a multi-turn conversation might have 20 short ones.

### 4.7.2 FLOP Analysis

Full causal attention computes S²/2 ≈ 33.6M attention entries per head. The span-materialized regime computes, for each span, (span_length) × (accumulated_cache_length + span_length/2) attention entries. For the reference working point (N=16, average L=512), this sums to ~35.7M entries per head. The ~6% difference is noise.

The key property: total FLOPs are at parity regardless of how spans are distributed within the sequence. A sequence with 4 spans of 2048 tokens and a sequence with 20 spans of ~400 tokens have the same total attention entries as a single 8192-token sequence. Cache materialization does not impose a compute tax. The potential overhead is from kernel launch latency and GPU under-utilization on small spans—but chain fusion and span-parallel batching (Section 6) largely eliminate these costs (Section 7.3).

### 4.7.3 Per-Step Overhead

A naive implementation—N sequential FlashAttention calls per layer—would incur kernel launch overhead (~30μs per call) and GPU under-utilization on short spans. But as described in Section 6.4, the span algebra admits two optimizations that eliminate most of this cost:

**Chain fusion.** When spans form a linear chain, the attention is mathematically identical to standard causal attention over the full sequence. One kernel launch; slice K,V at span boundaries afterward. **Zero kernel launch overhead for chains.**

**Span-parallel batching.** Independent spans are batched into a single `varlen` FlashAttention call. One kernel launch per wave, not per span.

| DAG structure | Kernel launches per layer | Overhead |
|---|---|---|
| Full chain (all spans sequential) | 1 (fused) | ~0% |
| RAG: system → 5 parallel docs → turn → response | 4 (one per wave) | Minimal |
| Fully independent spans (degenerate) | 1 (all batched in one wave) | ~0% |
| Mixed: some chains, some parallel | Waves in DAG | Data-dependent, low |

The residual overhead is from cache write operations (concatenation, optional transforms), which are memory-bandwidth-bound and small relative to attention compute.

**Net per-step overhead: ~1.0–1.1×** for chain-dominated data. **~1.05–1.15×** for mixed DAGs.

### 4.7.4 Memory

The KV cache across all spans and layers is the main incremental cost. For the reference configuration:

```
Cache size = S × d_model × 2 (K and V) × 2 (bf16 bytes) × 32 (layers)
           = 8192 × 4096 × 2 × 2 × 32
           ≈ 8.6 GB
```

This is the maximum cache size at the final span (which attends to all previous spans' caches). The total is determined by the sequence length S, not by the span decomposition—the same tokens are cached regardless of how they're partitioned. Against ~16GB of model parameters and ~48GB of optimizer state, this is modest.

With span-boundary gradient checkpointing, only the cache and span inputs are retained. Within-span activations are recomputed during backward at the cost of ~30% additional compute. Total memory overhead for the cache: **~8–10GB per training replica**, well within the headroom of an H100 (80GB) or GB200 (192GB).

### 4.7.5 Wall-Clock Estimates

**Phase 2 token budget: 100B tokens on an 8B model.**

Total FLOPs: 6 × 8×10⁹ × 100×10⁹ = 4.8×10²¹

| Hardware | Config | Effective TFLOPS | Baseline | With ~1.1× overhead |
|---|---|---|---|---|
| 8× H100 SXM | TP=4, FSDP=2 | ~3,560 | ~15.6 hrs | ~17 hrs |
| 72× GB200 NVL72 | TP=8, FSDP=9 | ~81,000 | ~16 hrs | ~17.5 hrs |

5B-token validation runs for quick iteration:

| Hardware | Time |
|---|---|
| 8× H100 | ~1.3 hours |
| 72× GB200 | ~6 minutes |

### 4.7.6 Total Training Budget Impact

| Component | Estimate |
|---|---|
| Phase 1 tokens (standard) | ~1.8–1.9T |
| Phase 2 tokens (cache-materialized) | ~100–200B (5–10% of total) |
| Phase 2 per-step overhead | ~1.05–1.15× (with chain fusion + span parallelism) |
| **Net wall-clock increase on total training** | **~0.5–1.5%** |

## 4.8 Implementation on LMEngine

### 4.8.1 Existing Infrastructure (No Changes)

**FSDP / ZeRO.** Cache materialization is sequence-local. Each data-parallel replica independently processes its spans and maintains its own cache. The all-reduce over gradients at step boundaries is unchanged. FSDP shards parameters and optimizer state, not activation-level cache tensors.

**Tensor Parallelism.** TP shards attention heads across devices. Cache entries shard by head: the device holding heads [0..H/P) stores corresponding K,V cache slices. Cross-span attention is standard multi-head attention from the TP perspective—the same all-reduce pattern applies. Cache write transforms should operate per-head or per-head-group to avoid introducing new cross-device communication.

**Optimizer and LR Scheduler.** Unchanged. Cache-materialized training doesn't alter the number of tokens per step, only the processing order within a step.

**FlashAttention / XMA Kernels.** Within-span attention is a standard causal FlashAttention call. Cross-span attention is a non-causal call with the accumulated cache as the KV input. FlashAttention's variable-length (`varlen`) interface handles both, including variable-length spans.

### 4.8.2 Adapting Existing Infrastructure (Minimal Changes)

**Padding-free transformers.** LM Engine already processes variable-length sequences as unpadded lists, eliminating wasted computation on pad tokens. Span-aware training extends this naturally: a sequence becomes an ordered list of variable-length spans, each an unpadded list. The within-span forward pass is structurally identical to what padding-free transformers already do. The variable-length span structure is a direct fit—no padding or bucketing of spans to uniform length.

**Block-level gradient checkpointing.** LM Engine supports `gradient_checkpointing_method: block` with `checkpoint_every_n_layers`. Cache-materialized training adds a span-boundary axis on top. The two axes are orthogonal and compose: layer-level checkpointing within a span, span-level checkpointing across spans.

### 4.8.3 New Implementation

**Component 1: `forward_span()` method.**

The model's forward method is extended to accept and return a KV cache dictionary. Each attention layer reads from the cache for cross-span context and writes its K,V outputs back. Spans are variable-length; the method imposes no length constraint.

```python
class GraniteForCausalLM(nn.Module):
    
    def forward_span(self, input_ids, kv_cache=None, cache_transform=None):
        """
        Process a single span (variable-length) with cross-span attention
        from kv_cache. Returns logits and updated kv_cache (in the
        computation graph).
        """
        hidden = self.embed(input_ids)
        new_cache = {}
        
        for i, layer in enumerate(self.layers):
            q, k, v = layer.attention.qkv_proj(hidden)
            
            # Build full KV: current span + accumulated cache
            if kv_cache and i in kv_cache:
                k_full = torch.cat([kv_cache[i][0], k], dim=-2)
                v_full = torch.cat([kv_cache[i][1], v], dim=-2)
            else:
                k_full, v_full = k, v
            
            # Single FlashAttention call (causal within span, full to cache)
            attn_out = flash_attention_varlen(q, k_full, v_full, causal=True)
            
            # Differentiable cache write
            if cache_transform is not None:
                k_write, v_write = cache_transform[i](k, v)
            else:
                k_write, v_write = k, v
            
            if kv_cache and i in kv_cache:
                new_cache[i] = (
                    torch.cat([kv_cache[i][0], k_write], dim=-2),
                    torch.cat([kv_cache[i][1], v_write], dim=-2),
                )
            else:
                new_cache[i] = (k_write, v_write)
            
            hidden = layer.ffn(attn_out)
        
        logits = self.lm_head(hidden)
        return logits, new_cache
    
    def forward_wave(self, span_list, kv_cache=None, cache_transform=None):
        """
        Process a wave of independent spans. Each span attends to the
        shared kv_cache but not to other spans in the wave.
        
        Production implementation: batch all spans into a single fused
        flash_attention_varlen call using cu_seqlens to demarcate span
        boundaries — the same mechanism LM Engine uses for padding-free
        transformers. This avoids N kernel launches for a wave of width N.
        """
        # Concatenate all spans into a single padded-free batch
        all_ids = torch.cat(span_list, dim=-1)
        cu_seqlens = compute_cu_seqlens(span_list)
        
        # Single fused forward pass over the batch
        logits, batch_cache = self.forward_span_batched(
            all_ids, cu_seqlens, kv_cache, cache_transform
        )
        
        # Split outputs and compose caches
        per_span_logits = split_by_cu_seqlens(logits, cu_seqlens)
        new_cache = compose_caches(kv_cache, batch_cache)
        return per_span_logits, new_cache
```

**Component 2: Span-sequential training loop.**

A wrapper around the existing training step that reads span boundary annotations and the dependency DAG from the data, then processes spans in topological waves. Independent spans within a wave are batched into a single kernel call.

```python
def topological_waves(span_deps):
    """Given span_deps[i] = list of predecessor indices,
    yield waves of span indices that can execute in parallel."""
    remaining = set(range(len(span_deps)))
    completed = set()
    while remaining:
        wave = [i for i in remaining 
                if all(d in completed for d in span_deps[i])]
        yield wave
        completed.update(wave)
        remaining -= set(wave)


def cache_materialized_training_step(model, batch, span_config, optimizer):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    span_boundaries = batch["span_boundaries"]  # from data pipeline
    span_deps = batch["span_deps"]              # dependency DAG
    
    cache = None
    total_loss = 0.0
    num_tokens = 0
    
    for wave in topological_waves(span_deps):
        # Collect all spans in this wave
        wave_ids = [input_ids[:, s:e] for s, e in 
                    [span_boundaries[i] for i in wave]]
        wave_labels = [labels[:, s:e] for s, e in 
                       [span_boundaries[i] for i in wave]]
        
        # Process all spans in wave (batched for GPU utilization)
        if span_config.checkpoint_spans:
            wave_logits, cache = checkpoint(
                model.forward_wave, wave_ids, cache,
                span_config.cache_transform,
                use_reentrant=False
            )
        else:
            wave_logits, cache = model.forward_wave(
                wave_ids, cache, span_config.cache_transform
            )
        
        for logits, span_labels in zip(wave_logits, wave_labels):
            mask = span_labels != -100
            span_loss = F.cross_entropy(
                logits[mask], span_labels[mask], reduction="sum"
            )
            total_loss += span_loss
            num_tokens += mask.sum()
    
    loss = total_loss / num_tokens
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

When the DAG is a chain (every span depends on all prior spans), each wave has width 1 and this degenerates to purely sequential processing. No special-casing is needed. However, the chain fusion optimization (Section 6.4) can be applied: if the full sequence is a chain, the training loop can bypass the wave decomposition entirely and run a single fused FlashAttention call over the full sequence, slicing K,V outputs at span boundaries afterward. This is a performance optimization, not a correctness concern—the wave-based loop produces identical results.

**Component 3 (optional): Cache write transform modules.**

These modules implement the optional learned transforms described in Section 5.3. They are not required for the core cache materialization—the system works with identity writes and no new parameters. They are included here because the implementation is straightforward once the `forward_span` / `forward_wave` machinery exists, and because they illustrate the extensibility of the cache write path.

Pluggable, per-layer modules registered as part of the model so FSDP handles their parameters automatically. Because spans are variable-length, these modules must handle variable input sizes. The compressor uses cross-attention with a fixed number of learned queries, producing a fixed-size output regardless of input span length—a natural fit for variable-length semantic units. The evictor retains a fraction of entries, so its output length scales with input length.

```python
class CacheCompressor(nn.Module):
    """
    Learned compression via cross-attention with learned queries.
    Produces a fixed number of cache entries per span regardless
    of input span length — a 2000-token document and a 50-token
    turn both compress to num_summary_tokens entries.
    """
    
    def __init__(self, num_summary_tokens, d_head, num_heads):
        super().__init__()
        self.queries = nn.Parameter(
            torch.randn(num_heads, num_summary_tokens, d_head) * 0.01
        )
    
    def forward(self, k, v):
        B, H, L, D = k.shape
        q = self.queries.unsqueeze(0).expand(B, -1, -1, -1)
        # Per-head cross-attention: queries attend to span's K,V
        # Produces num_summary_tokens compressed entries per head,
        # regardless of L
        k_out = torch.einsum('bhqd,bhkd->bhqk', q, k).softmax(-1) @ k
        v_out = torch.einsum('bhqd,bhkd->bhqk', q, v).softmax(-1) @ v
        return k_out, v_out


class CacheEvictor(nn.Module):
    """Learned per-entry importance scoring with differentiable selection."""
    
    def __init__(self, d_head, retain_fraction=0.5, temperature=1.0):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_head, d_head // 4),
            nn.GELU(),
            nn.Linear(d_head // 4, 1),
        )
        self.retain_fraction = retain_fraction
        self.temperature = temperature
    
    def forward(self, k, v):
        scores = self.scorer(k).squeeze(-1)  # (B, H, L)
        retain_k = max(1, int(k.shape[-2] * self.retain_fraction))
        
        # Straight-through Gumbel softmax for differentiable top-k
        weights = F.gumbel_softmax(
            scores, tau=self.temperature, hard=False, dim=-1
        )
        _, topk_idx = weights.topk(retain_k, dim=-1)
        topk_idx = topk_idx.sort(dim=-1).values
        
        k_out = torch.gather(
            k, -2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1])
        )
        v_out = torch.gather(
            v, -2, topk_idx.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
        )
        return k_out, v_out
```

**Component 4: Data pipeline — span boundary and dependency extraction.**

The mid-training data pipeline extracts span boundary positions and the dependency DAG from the LLMON structure of each training sequence. LLMON uses special tokens to delimit structured objects (documents, turns, retrieval results, etc.); the boundary extractor identifies these delimiters, emits an array of (start, end) token positions per sequence, and derives the dependency graph from the LLMON object hierarchy. Sibling objects (e.g., multiple documents within a retrieval result container) are independent; a child object depends on its parent's cache. This is a preprocessing step—boundaries and dependencies are determined once when the data is tokenized.

For LM Engine's Megatron-LM-based dataloader, this means extending the tokenized data format to include `span_boundaries`, `span_types`, and `span_deps` fields per sequence.

```yaml
# Data format extension
# Each training example includes:
#   input_ids: [token_ids...]   # includes LLMON special tokens
#   labels: [label_ids...]
#   span_boundaries: [[0, 487], [487, 1203], [1203, 2891], [2891, 3402], [3402, 8192]]
#   span_types: ["system", "document", "document", "turn", "response"]
#   span_deps: [[], [0], [0], [0, 1, 2], [0, 1, 2, 3]]
#
# span_deps[i] = indices of spans whose caches span i attends to.
# Sibling documents [1] and [2] both depend on system [0] but not
# on each other — they can be processed in parallel.
```

**Component 5: Configuration.**

```yaml
span_training:
  enabled: true
  checkpoint_spans: true
  min_span_length: 32              # coalesce spans shorter than this
  
  # Optional: learned cache transforms (default: identity / disabled)
  cache_transform:
    type: "identity"               # "identity" | "compress" | "evict" | "pipeline"
    compression:                   # only used if type includes compression
      num_summary_tokens: 64       # fixed output size per span
      warmup_steps: 2000           # anneal from identity to full compression
    eviction:                      # only used if type includes eviction
      retain_fraction: 0.5
      initial_temperature: 5.0
      final_temperature: 0.1
      anneal_steps: 5000
```

The default configuration (`type: "identity"`) runs the core proposal with no learned transforms and no new parameters.

### 4.8.4 Phase Transition: Checkpoint Handling

For the core proposal (identity writes), no new parameters are introduced. The Phase 1 checkpoint is loaded as-is. The only change is the training loop structure—span-sequential processing instead of monolithic forward pass.

If optional learned transforms are enabled in later stages, the checkpoint is extended with `CacheWriteTransform` parameters. New parameters are initialized near-identity:

- Compressor queries initialized with small random values so initial output approximates a uniform average (near-identity for the compression operation).
- Evictor scorer initialized with near-zero weights so all entries receive similar scores (retain-all bias).

LM Engine's checkpoint loading handles this as a `strict=False` load followed by explicit initialization of new parameters.

### 4.8.5 Multi-Accelerator Notes

**CUDA (NVIDIA).** Primary target. FlashAttention 2's `varlen` interface handles variable-length spans natively.

## 4.9 Curriculum Within Phase 2

Phase 2 has a single mandatory stage and two optional stages. The mandatory stage validates the core proposal. The optional stages explore the learned cache transform extensions, and should only be pursued after the core is validated.

**Stage 2a — Core: Identity cache writes (10–20B tokens).** This is the core proposal. The model adapts to span-sequential processing with identity cache writes—no new parameters, no learned transforms. This stage validates overhead estimates, confirms loss recovery after the Phase 1 transition, and establishes whether cache materialization alone (without learned transforms) improves span composition fidelity and narrows the train/serve gap. If Stage 2a shows clear benefits, it may be sufficient on its own.

**Stage 2b — Optional: Compression (30–50B tokens).** Enable `CacheCompressor`. Initialize near-identity, anneal toward full compression. The compressor's parameters receive gradients from the first step; the annealing schedule controls how aggressively it compresses. The compressor produces a fixed number of summary tokens per span regardless of span length, so short turns and long documents are compressed to the same cache footprint—the model learns whether a 50-token turn and a 2000-token document warrant the same summary budget. Only pursue if there is a concrete need for reduced cache size at serving time.

**Stage 2c — Optional: Eviction (20–30B tokens).** Enable `CacheEvictor` alongside compression. Initialize with high Gumbel temperature (soft, retain-all), anneal toward hard selection. The evictor learns which tokens to retain given that survivors will also be compressed. Only pursue if compression alone doesn't provide sufficient cache budget reduction, or if eviction patterns are of independent research interest.

**Stage 2d — Evaluation and iteration.** Measure span composition fidelity, compression quality at fixed cache budgets, and downstream task performance. Iterate on parameters and curriculum schedule.

Token budgets are approximate and should be validated by monitoring loss curves. If loss plateaus before the budget is exhausted, advance to the next stage. If loss hasn't recovered, extend the budget.

## 4.10 Evaluation

### Core (Stage 2a)

**Span composition fidelity.** Given independently processed spans A, B, C (e.g., three documents), does compose(A, compose(B, C)) produce the same downstream behavior as processing the full concatenation A‖B‖C? Measure divergence in output distributions. Cache-materialized training should reduce this divergence because the model has trained with composed caches. This is the primary success metric for the core proposal.

**Train/serve gap.** Compare the model's behavior when served with an actual KV cache (the inference regime) against its behavior during full-attention evaluation (the training regime). Cache-materialized training should narrow this gap.

**Overhead validation.** Confirm the ~1.05–1.15× per-step overhead estimate (with chain fusion and span parallelism) across the actual data mixture. Measure kernel launch counts per step to verify that chain fusion and span-parallel batching are activating as expected.

### Extensions (Stages 2b, 2c)

**Quality at fixed cache size.** Compare perplexity and task accuracy between (a) a baseline model with cache truncation to size C, and (b) a model with learned compression/eviction to size C. The learned approach should degrade more gracefully as C decreases. Only meaningful if learned transforms are pursued.

**Eviction selectivity.** Inspect which tokens the learned evictor retains. Does the policy correlate with known importance heuristics (attention sinks, entity mentions, reasoning pivots)? Does it discover novel retention patterns? Do retention patterns differ by span type (document vs. turn vs. instruction)?

## 4.11 Risks and Mitigations

**Overhead exceeds estimates (core).** The ~1.05–1.15× estimate assumes chain fusion and span-parallel batching are effective across the data mixture. If the data contains structures that defeat both optimizations (neither chainable nor parallelizable), the overhead could approach the naive 1.2–1.3× estimate. Mitigation: the 5B-token validation run (Stage 2a) measures this across the actual data mixture before committing to the full Phase 2 budget.

**Loss doesn't recover after Phase 1 → Phase 2 transition (core).** The distribution shift from full attention to span-sequential attention may cause a loss spike that doesn't recover. Mitigation: gradual transition (linearly increase the fraction of cache-materialized batches over a warmup window of 1–2B tokens). If the model is fundamentally unable to adapt, this indicates a deeper issue with span-sequential processing, which is valuable to know early.

**Very short spans degrade GPU utilization (core).** Conversational data with many brief turns (20–50 tokens each) could in principle under-saturate the GPU. In practice, chain fusion eliminates this for sequential conversations (the entire chain runs as a single fused kernel), and span-parallel batching handles independent short spans. The remaining risk is data with many short spans that are neither chainable nor parallelizable—an unusual structure. Mitigation: the data pipeline can coalesce consecutive very short spans (below `min_span_length`) into their successor if needed.

**Gradient vanishing through long cache chains (core).** With many spans, gradients must flow through many cache write operations to reach early spans. For sequences with N=20 spans, this could cause vanishing gradients. Mitigation: truncate gradient flow to a window of K recent caches (e.g., K=4 or K=8). Ablate to find the minimum K that preserves span composition quality.

**Compressor capacity across span lengths (extension only).** A fixed number of summary tokens (e.g., 64) may be too few for a 2000-token document and too many for a 20-token turn. Mitigation: make `num_summary_tokens` a function of span length (e.g., proportional with a floor and ceiling), or let the evictor prune before compression so the compressor always sees a manageable input. Only relevant if learned compression is pursued.

## 4.12 Open Questions

**Core:**

1. **Optimal Phase 2 token budget.** Is 5% of total tokens sufficient for Stage 2a (identity writes), or does cache-awareness require closer to 10%? The staged curriculum provides natural checkpoints for evaluating whether additional tokens are yielding improvements.

2. **Gradient flow depth.** Should gradients flow through all N span caches, or is truncated backpropagation through a window of K recent caches sufficient? Longer gradient paths are more expressive but risk vanishing gradients and increase memory.

3. **DAG depth distribution in mid-training data.** The benefit of span parallelism depends on the width of the dependency DAGs in the data mixture. What fraction of mid-training sequences have DAG width > 1? Profiling the actual data will determine whether span parallelism is a minor optimization or a major overhead reduction.

**Extensions (only relevant if pursuing learned transforms):**

4. **Interaction with GQA.** Grouped-query attention already reduces cache size by sharing K,V heads. Does learned compression on top of GQA compound the benefit, or do they serve the same purpose? Ablation needed.

5. **Associativity of learned merge.** If the merge operator replaces concatenation, is approximate associativity (learned, not enforced) sufficient for the span algebra? If strict associativity is required, what architectural constraints on the merge operator guarantee it?

6. **Fixed vs. adaptive summary budget.** Should every span, regardless of length, compress to the same number of summary tokens? Or should longer spans get proportionally more summary tokens? The former is simpler and produces predictable cache sizes; the latter may preserve more information from long spans. The right answer likely depends on the data mixture and on the semantic density of different span types.

7. **Span-type-aware transforms.** Should the cache write transform be conditioned on span type (document, turn, instruction, retrieval result)? A system prompt may warrant different compression than a retrieved passage. This could be implemented as a routing mechanism over a small family of specialized compressors, with span type derived from the LLMON object structure driving the routing. The `span_types` field in the data format (see Component 4) already provides this metadata.

8. **Commutativity of merge under span parallelism.** If independent spans in a wave write to a shared cache via a learned merge operator, the merge must be commutative (order-independent) for correctness. Concatenation is commutative for independent spans. Is this a hard requirement for learned merge, and if so, what architectural constraints enforce it?

## 4.13 Out of Scope

This document covers single-node cache-materialized training for dense transformer architectures. The following are recognized as important but are deferred to separate design work:

**Pipeline parallelism.** Cache-materialized training introduces a 2D dependency grid (pipeline stages × spans) that requires a wavefront schedule to replace the standard 1F1B pipeline. This is the hardest distributed systems task and is unnecessary for models that fit within a single TP+FSDP node—up to ~13B on 8× H100, up to ~65–70B on GB200 NVL72. The wavefront schedule should be designed and implemented after the core algorithm is validated in the PP-free regime.

**Cross-Layer Attention (CLA).** CLA shares KV caches across groups of layers, reducing the number of distinct caches. This interacts with cache-materialized training (shared caches mean shared cache transforms), but the interaction is straightforward and can be addressed as an incremental extension.

**Mixture-of-Experts (MoE).** Expert routing in MoE architectures interacts with span boundaries if routing decisions depend on or affect cache structure. For standard MoE where expert routing is in the FFN block (downstream of attention), the interaction is minimal. For architectures where expert routing affects attention, the interaction needs analysis.

**Ring Attention / sequence parallelism.** Ring Attention distributes sequence chunks across devices in a ring topology, which maps naturally onto span-sequential processing. This is a promising direction for scaling cache-materialized training to very long sequences across multiple nodes, but is deferred until the single-node algorithm is validated.

---

> [← Ch.3: Model-Runtime Codesign](03-model-runtime-codesign.md) · [Table of Contents](../README.md) · [Ch.5: Serving Engine →](05-serving-engine.md)
