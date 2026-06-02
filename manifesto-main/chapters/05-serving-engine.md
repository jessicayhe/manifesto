> [← Ch.4: Cache-Materialized Training](04-cache-materialized-training.md) · [Table of Contents](../README.md) · [Ch.6: Span Attention →](06-span-attention.md)

---

# Chapter 5: Serving Engine

**Component:** `mellea-core`
**Status:** Draft
**Version:** 0.1
**Authors:** David Cox, Claude
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-06
**Scope:** NoPE / DRoPE transformer models. Granite is the primary target.

**Prerequisites:** Chapter 1 (Span Algebra — free monoid, fan-in composition, provenance/taint), Chapter 2 (LLMON §2.5 — boundary-constrained masking), Chapter 3 (Model-Runtime Codesign §3.5–3.6)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## 5.1 Position Encodings Considered Harmful

### 5.1.1 The Position Encoding Tax

Standard transformers bake absolute position into every key vector via Rotary Position Embedding (RoPE). The key for token "the" at position 5 is a *different vector* than the key for "the" at position 12 — they've been rotated by different angles in the complex plane. This makes the KV cache position-dependent: a cached key can only be reused at the exact position where it was computed.

This position dependence imposes a structural constraint on KV cache composition. Given two independently computed KV segments A (computed at positions 0-4) and B (computed at positions 0-3), you cannot simply concatenate them and use the result as a prefix for generation, because B's keys encode the wrong positions — they should be at positions 5-8, not 0-3. You have two options: re-prefill B at the correct positions (expensive), or restrict yourself to only composing segments that were originally computed in a compatible positional sequence (limiting).

Formally, with RoPE, the set of KV cache segments under composition forms a [partial order](https://en.wikipedia.org/wiki/Partially_ordered_set). Only segments whose positional provenance forms a compatible chain can be composed cheaply. All other compositions require recomputation of some sort.

### 5.1.2 NoPE Enables Free Monoid Composition

Models without positional encoding — NoPE (No Position Encoding) or DRoPE (Dispensing with RoPE) — produce keys and values that are pure content projections: `K = x @ W_k`, with no position-dependent rotation. The key for "the" is the same vector regardless of where it appears in the sequence. Positional structure comes entirely from the causal attention mask, not from the key/value representations.

This changes the algebraic structure of composition. Without position encoding, any set of KV segments can be concatenated in any order at zero cost. There is no "wrong position" because there are no positions baked into the cached values. The set of KV segments under concatenation forms a [free monoid](https://en.wikipedia.org/wiki/Free_monoid): composition is always valid, always free, and the only constraint is the attention mask that determines visibility.

This enables a qualitative change in what the serving infrastructure can express:

| Operation | RoPE | NoPE |
|---|---|---|
| Sequential prefix reuse | ✓ (radix cache) | ✓ |
| Fan-in (merge independent segments) | Re-prefill required | Free concatenation |
| Reorder segments | Re-prefill required | Free (mask determines order) |
| Share segment across different compositions | Only if positions align | Always |
| Compose at the memory level | Tensor copy or re-prefill | Integer list concatenation |

Empirically, hybrid models can be trained with NoPE full attention layers (Granite 4-h being a prime example), and models that there trained with RoPE can drop it after training (using the DRoPE technique from Sakana AI; we have successful converted Granite 4.1 RoPE checkpoints to NoPE). This design document assumes a NoPE model. All subsequent design decisions — free composition, position-independent slot allocation, scope-based memory management — depend on this property. A RoPE model would require a significantly different (and more constrained) serving architecture.

---

## 5.2 Core Principle: The Span Algebra is the Memory Manager

### 5.2.1 Problem with Existing Engines

Existing LLM serving engines (vLLM, SGLang) treat the KV cache as an implementation detail of attention. Memory management (paged allocation, prefix caching, eviction) operates on opaque token-position pairs. The serving engine doesn't know *why* tokens are shared across requests — it discovers shared prefixes via byte-level token sequence matching.

This opacity prevents three operations that the Mellea span algebra requires:

**Scope isolation.** Two requests may include the same document span but with different visibility rules — one can attend to it, one cannot. An opaque cache has no mechanism to express per-request visibility over shared KV.

**Taint breaking.** A span's KV was computed attending to sensitive context. The tokens are still useful, but the KV needs to be recomputed under a clean (empty) scope to sever the information channel. An opaque cache has no concept of provenance — it doesn't know *what context* a cached KV was computed under.

**Free composition.** In a NoPE model, any set of independently computed KV segments can be concatenated without recomputation. An opaque cache that keys on token-sequence prefixes will treat different orderings of the same segments as different prefixes, missing the reuse opportunity.

### 5.2.2 Solution: Spans Own Slots, Not Tensors

In this design, the span registry — names, tags, scopes, provenance — is not a metadata layer sitting above the cache. It IS the cache. Each span holds a list of integer slot indices into a pre-allocated KV pool. Composing spans means concatenating integer lists. Sharing spans across requests means multiple requests referencing the same slot indices. Evicting a span means returning its slot indices to the free list. The KV pool is the only large allocation; everything else is bookkeeping over integer indices.

---

## 5.3 Concepts

**Span.** A named segment of tokens with metadata: a unique name, a semantic tag (INSTRUCT, DATA, EXEC, SENSITIVE, ...), a scope (which other spans were visible when this span's KV was computed), and a list of slot indices into the KV pool. Spans are the atoms of memory management.

**Scope.** The ordered list of span names visible during a fill or gen operation. Determines the attention mask: a token can attend to all KV slots belonging to spans in its scope. The scope is declared by the caller, not discovered by the runtime.

**Fill.** Prefill a span's tokens under a given scope: run the tokens through the model (attending to the scope's KV), write the resulting K/V projections into freshly allocated pool slots, register the span.

**Gen.** Autoregressive generation attending to a scope of spans. Each decode step produces one token, writes its KV into one new pool slot, and samples the next token.

**Gen_parallel.** Batch generation where multiple requests share a common scope. The scope IS the batch definition — the caller declares what's shared. Shared prefix KV is gathered once; per-request suffix KV is gathered per-request; attention over the two is merged via cascade attention (Section 6.4). No prefix detection heuristic.

**Refill.** Taint-breaking primitive. Discard a span's pool slots and re-prefill its tokens under a new (typically empty) scope, severing attention-mediated information flow from the original context. Copy-on-write protects active sequences that still reference the old KV.

**Compose.** Merge multiple spans into one by concatenating their slot index lists. Because the model is NoPE, this is pure integer list concatenation — zero compute, zero tensor operations. The composed span's KV is the union of its components' KV, immediately usable in any scope.

---

## 5.4 Architecture

```
┌─────────────────────────────────────────────┐
│  Clients (HTTP / gRPC / in-process)         │
│    POST /gen_parallel {scope, ops}          │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  Request Queue                              │
│    Pending requests with scope metadata     │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  Scheduler                                  │
│    Scope-aware admission policy             │
│    Preemption / eviction decisions          │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  Batch Executor                             │
│    Batched forward pass (prefill + decode)  │
│    Gather-based KV access from pool         │
│    Scope-grouped cascade attention          │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  KV Pool + Span Registry                   │
│    Pre-allocated pool tensors               │
│    Span → slot index mapping                │
│    Ref-counted sharing, LRU eviction        │
│    Free-monoid composition (NoPE)           │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│  Model (via backend abstraction)            │
└─────────────────────────────────────────────┘
```

---

## 5.5 KV Pool

### 5.5.1 Layout

Pre-allocate a fixed-size pool at startup:

```
k_pool[layer]: (max_tokens, n_kv_heads, head_dim)   float16/bfloat16
v_pool[layer]: (max_tokens, n_kv_heads, head_dim)
```

For Granite Micro (40 layers, 8 KV heads, 64 head_dim, bfloat16):
- Per-token cost: 40 × 2 × 8 × 64 × 2 bytes = 81,920 bytes ≈ 80 KB
- 16K token pool: ~1.25 GB
- 64K token pool: ~5 GB

Pool size is the primary memory budget knob.

### 5.5.2 Slot Allocator

```python
class SlotAllocator:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.free = list(range(max_tokens))

    def alloc(self, n: int) -> list[int]:
        if len(self.free) < n:
            raise OOMError(f"Need {n} slots, have {len(self.free)}")
        slots = self.free[:n]
        self.free = self.free[n:]
        return slots

    def release(self, slots: list[int]):
        self.free.extend(slots)
```

Slots are allocated per-span. A span "sys" with 100 tokens holds 100 slot indices. Because the model is NoPE, slot ordering within a span is semantically irrelevant — the KV at slot 7 and slot 2043 are equally valid positions for the same token's key/value pair. The gather-based attention (Section 6.1) reads slots by index regardless of physical layout.

This means pool fragmentation has no performance cost. After many alloc/free cycles, a span's slots may be scattered across the pool. For a RoPE model this would be a problem (positions would be wrong). For NoPE, the gather produces identical results regardless of physical slot arrangement.

### 5.5.3 Span

```python
@dataclass
class Span:
    name: str
    tag: str                        # INSTRUCT, DATA, EXEC, SENSITIVE, ...
    token_ids: list[int]
    slots: list[int]                # indices into the KV pool
    filled_scope: tuple[str, ...]   # provenance: which spans were visible during fill
    refs: int = 1                   # shared-prefix ref counting
    last_access: float = 0.0       # LRU eviction
```

### 5.5.4 Compose

```python
def compose(self, names, new_name):
    all_slots = []
    for n in names:
        all_slots.extend(self.spans[n].slots)
    self.spans[new_name] = Span(name=new_name, slots=all_slots, ...)
```

This is the free-monoid composition made concrete. `compose(["A", "B"], "AB")` and `compose(["B", "A"], "BA")` produce spans with the same pool content, just with slot indices in different order. Because the model is NoPE, both orderings produce identical attention output — the attention mask, not the slot order, determines the sequential relationship between tokens.

### 5.5.5 Ref-Counted Sharing and Eviction

Multiple active sequences may reference the same span. A span's slots are freed only when `refs` drops to zero and the span is selected for eviction. Eviction proceeds leaf-first: spans not referenced by any other span's `filled_scope` are candidates. Among candidates, the least-recently-accessed span is evicted first.

---

## 5.6 Batched Forward Pass

### 5.6.1 Gather-Based KV Access

The attention layer gathers KV from the pool using slot indices:

```python
def gather_kv(pool_k, pool_v, slot_indices):
    """
    pool_k:       (max_tokens, n_kv_heads, head_dim)
    slot_indices: (total_kv_len,) int tensor
    Returns:      (1, n_kv_heads, total_kv_len, head_dim)
    """
    k = pool_k[slot_indices]                  # (total_kv_len, n_kv, D)
    return k.permute(1, 0, 2).unsqueeze(0)    # (1, n_kv, S, D)
```

Because the model is NoPE, the gathered KV is correct regardless of the original slot layout — there's no position-dependent rotation to worry about. The gather is a pure content retrieval.

### 5.6.2 Write-Back for New Tokens

During decode, each sequence generates one new KV pair per layer. Write directly into the next allocated slot:

```python
new_slots = [seq.next_slot for seq in active_seqs]
pool_k[layer_idx][new_slots] = k_new_batch    # (B, n_kv, D)
pool_v[layer_idx][new_slots] = v_new_batch
```

### 5.6.3 Heterogeneous Sequence Lengths

Active sequences have different total KV lengths. Two approaches, used together:

**Padded batch** for sequences with different scopes. Pad all slot index lists to `max_kv_len`, use an attention mask to zero out padding. Wastes some compute but keeps the matmul rectangular.

**Scope-grouped cascade** for sequences sharing a scope. Group sequences by their scope, run cascade attention per group. This is the common case and the efficient path.

### 5.6.4 Cascade Attention (Shared-Scope Batching)

When B sequences share a scope and differ only in their suffix, attention decomposes into two passes over the shared and unique KV segments, merged via numerically stable log-sum-exp:

```python
def cascade_attention(q, k_shared, v_shared, k_unique, v_unique, scale):
    # Pass 1: shared prefix (gathered once, broadcast to all B sequences)
    s1 = (q @ k_shared.T) * scale               # (B, H, 1, S_shared)
    m1 = s1.max(axis=-1, keepdim=True)
    e1 = (s1 - m1).exp()
    d1 = e1.sum(axis=-1, keepdim=True)
    n1 = e1 @ v_shared

    # Pass 2: unique suffix (gathered per-sequence)
    s2 = (q @ k_unique.T) * scale                # (B, H, 1, S_suffix)
    m2 = s2.max(axis=-1, keepdim=True)
    e2 = (s2 - m2).exp()
    d2 = e2.sum(axis=-1, keepdim=True)
    n2 = e2 @ v_unique

    # Combine: log-sum-exp correction preserves numerical stability
    m = maximum(m1, m2)
    c1, c2 = (m1 - m).exp(), (m2 - m).exp()
    return (n1 * c1 + n2 * c2) / (d1 * c1 + d2 * c2)
```

The shared KV is `(1, H, S, D)` — gathered once from the pool using the scope's slot indices and broadcast across the batch dimension. Per-sequence suffix KV is `(B, H, U, D)`. All operations are standard matmul, exp, sum. No custom kernels.

This generalizes to N-way cascade for multi-level sharing (shared system prompt → shared document → unique query), using iterative log-sum-exp accumulation across segments.

The NoPE property is what makes this decomposition exact rather than approximate. With RoPE, the shared prefix KV would need position-corrected keys for each sequence's offset — different sequences at different total lengths would need different rotations applied to the shared keys. With NoPE, the shared keys are pure content projections, and the same gathered KV is correct for all sequences regardless of their total context length.

---

## 5.7 Request Lifecycle

```
PENDING ──admit──▶ PREFILLING ──done──▶ DECODING ──eos/max──▶ FINISHED
                        │                   │
                        ▼                   ▼
                   (writes KV          (writes 1 token
                    into pool           per step into
                    slots)              pool slot)
```

### 5.7.1 Request

```python
@dataclass
class Request:
    id: str
    scope: list[str]             # shared prefix span names
    suffix_tokens: list[int]     # unique suffix tokens
    gen_op: GenOp                # name, max_tokens, tag, sampling params
    status: str = "pending"
    generated: list[int] = field(default_factory=list)
    suffix_slots: list[int] = field(default_factory=list)
```

### 5.7.2 Admission

On each step, the scheduler considers pending requests:

1. Compute slots needed: `len(suffix_tokens) + max_new_tokens`.
2. Check if the scope spans' slots are already allocated. If so, only suffix + generation slots are needed — the shared KV is reused, not re-allocated.
3. If enough free slots: admit, allocate suffix slots, prefill suffix, transition to DECODING.
4. If not enough: wait, or preempt a lower-priority active sequence.

### 5.7.3 Preemption

When pool space is tight:
- Free finished sequences' non-shared suffix slots first.
- If still tight, evict LRU leaf spans.
- If still tight, preempt the lowest-priority active sequence: save its generated tokens, free its suffix slots, re-queue it. When readmitted, it re-prefills its suffix (scope KV is likely still allocated because it's shared).

---

## 5.8 Scheduler Policy

### 5.8.1 Scope-Aware Admission

The scheduler preferentially admits requests whose scope overlaps with already-allocated spans. This maximizes KV reuse.

```python
def score_request(self, req, allocated_spans):
    shared = set(req.scope) & allocated_spans
    reuse_tokens = sum(self.spans[s].seq_len for s in shared)
    new_tokens = len(req.suffix_tokens) + req.gen_op.max_tokens
    return reuse_tokens / (new_tokens + 1)
```

Requests that reuse existing scope KV get priority. This naturally groups together requests sharing the same system prompt, document context, or tool definitions — the dominant pattern in Granite tool-calling workloads.

### 5.8.2 Fairness

Pure scope-affinity can starve requests with unique scopes. An aging term prevents starvation:

```python
score = scope_affinity + alpha * (now - req.submitted_at)
```

### 5.8.3 Batch Size Limits

- `max_batch_decode`: maximum sequences in decode phase per step (bounded by GPU activation memory, typically 16-64 for a 3B model).
- `max_batch_prefill`: maximum tokens in prefill per step (bounded by compute, typically 512-2048).
- Chunked prefill: process a chunk of a prefill request's tokens, then run a decode step for all active sequences, then continue the prefill. This prevents long prefills from blocking decode latency.

---

## 5.9 Scope-Isolated Masking

For requests requiring scope isolation, the attention mask per sequence encodes which spans are visible:

```python
def build_scope_mask(seq, total_kv_len):
    mask = full((1, total_kv_len), -inf)
    for span_name in seq.visible_spans:
        span = self.spans[span_name]
        mask[0, span.slots] = 0.0
    return mask
```

This mask is constructed from span metadata. A span can be physically present in the KV pool — its slots occupied, its keys and values computed — but informationally invisible to a particular request because it's not in that request's scope. The mask enforces this without modifying the KV.

This is where the NoPE property is essential for correctness. With RoPE, masking out a span in the middle of a sequence would create a gap in the position encoding — the keys after the gap would have wrong positions. With NoPE, masking out a span simply removes its content from the attention, with no positional side effects. The remaining visible spans attend to each other exactly as if the masked span's tokens had never existed.

For sequences in the same cascade group (same scope), the mask is identical and shared. For sequences with different scopes, the padded-batch approach handles heterogeneous masks naturally.

---

## 5.10 Refill (Taint Breaking)

Refill severs attention-mediated information flow by discarding a span's KV and recomputing it under a new scope:

1. **Copy-on-write check.** If `refs > 1` (other active sequences reference this span), clone the slots first: allocate new slots, copy KV, point existing users at the clone. This prevents disrupting active sequences.
2. Free the original slots.
3. Allocate fresh slots.
4. Re-prefill the span's tokens under the new scope (typically empty), writing new KV into fresh slots.
5. Update the span's slot list and `filled_scope`.

The NoPE property simplifies step 4: the re-prefilled KV is valid at any slot position. There's no need to ensure the new slots occupy the same position range as the old ones. The allocator can place them anywhere in the pool.

---

## 5.11 API

### 5.11.1 Synchronous

```python
class MelleaEngine:
    def fill(self, name, token_ids, scope=None, tag="DATA") -> Span
    def gen(self, name, scope, max_tokens=128, **sampling) -> GenResult
    def gen_parallel(self, scope, ops, **sampling) -> list[GenResult]
    def refill(self, name, scope=None) -> Span
    def compose(self, names, new_name, tag="DATA") -> Span
    def drop(self, name)
    def execute(self, program: list) -> dict[str, GenResult]
```

`gen_parallel` takes a `scope` (the shared prefix) and a list of `GenOp` (per-sequence suffixes). The scope IS the batch — the caller declares what's shared.

### 5.11.2 Async

```python
class MelleaServer(MelleaEngine):
    async def submit(self, request: Request) -> str
    async def result(self, request_id: str) -> GenResult
    def run(self)
```

The synchronous API is implemented on top of the async primitives. Programs written against the synchronous API work unchanged.

### 5.11.3 Program Execution

```python
engine.execute([
    {"op": "fill", "name": "sys", "tokens": [...], "tag": "INSTRUCT"},
    {"op": "fill", "name": "ctx", "tokens": [...], "scope": ["sys"]},
    {"op": "gen_parallel", "scope": ["sys", "ctx"], "ops": [
        {"name": "summary",   "suffix": [...], "max_tokens": 256},
        {"name": "entities",  "suffix": [...], "max_tokens": 128},
    ], "temperature": 0.0},
    {"op": "refill", "name": "summary", "scope": []},
])
```

---

## 5.12 Implementation Plan

### Phase 1: KV Pool + Gather Attention

- `KVPool` class with slot allocator.
- `Span` with `slots: list[int]`.
- Modified attention that gathers K/V from pool using slot indices.
- **Validation:** pooled forward produces numerically identical output to non-pooled forward.

### Phase 2: Batch Executor

- Batched decode with padded slot indices and per-sequence masks.
- Scope-grouped cascade attention.
- **Validation:** batched decode produces identical per-sequence output to sequential single-sequence decode.

### Phase 3: Request Queue + Scheduler

- Request lifecycle state machine.
- FCFS admission with scope-affinity scoring.
- Basic preemption.
- **Validation:** multiple concurrent requests with overlapping scopes complete correctly.

### Phase 4: Serving Loop

- Async submit/result API.
- Chunked prefill interleaved with decode.
- **Validation:** sustained throughput under concurrent load.

### Phase 5: Advanced

- Copy-on-write refill.
- Scope-isolated masking in batched attention.
- Fairness-adjusted scheduling.
- Pool defragmentation (optional — NoPE makes fragmentation costless, but free-list compaction may improve allocator performance for bookkeeping reasons).

---

## 5.13 Open Questions

**Multi-GPU.** Tensor-parallel attention requires the KV pool to be sharded across devices. Each device holds `n_kv_heads / n_devices` heads of the pool. The slot allocator is global; the gather is device-local.

**Quantized KV.** Storing KV in INT8 or FP8 doubles the effective pool size. The gather + dequantize path should be fused for performance.

**Hybrid models (Mamba + Attention).** Mamba-2 layers don't use KV — they maintain a fixed-size state tensor per sequence that is updated in-place. The pool needs a separate `state_pool` for Mamba states. Refill for Mamba layers requires re-running the full recurrence (the state at position N encodes the entire history 0..N).

**RoPE fallback.** If a RoPE model must be supported, the span registry would need to track positional provenance and enforce compatible-chain validation before composition. Compose would require re-prefilling. The slot allocator would need to track position assignments. All of these are the constraints that NoPE eliminates — they can be added as a compatibility mode but at significant complexity cost.

---

## 5.14 Estimated Scope

| Component | Lines | Difficulty |
|---|---|---|
| KVPool + SlotAllocator | ~150 | Straightforward |
| Gather-based attention | ~100 | Medium (scatter/gather perf) |
| Pooled span registry | ~100 | Straightforward |
| Cascade grouping + attention | ~120 | Medium (log-sum-exp math) |
| Request lifecycle | ~80 | Straightforward |
| Scheduler (FCFS + scope affinity) | ~120 | Medium (policy tuning) |
| Serving loop | ~100 | Straightforward |
| **Total new code** | **~770** | |

Plus ~200 lines of tests per phase.

---

> [← Ch.4: Cache-Materialized Training](04-cache-materialized-training.md) · [Table of Contents](../README.md) · [Ch.6: Span Attention →](06-span-attention.md)
