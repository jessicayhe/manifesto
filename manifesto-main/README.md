# Generative Computing

**Programming Models, Abstractions, and Tooling for LLM-Based Software**

*David Cox and the AI Foundations Team · April 2026*

**Version:** 1.2
**Last Updated:** 2026-04-06

---

"Generative computing" refers to the idea that language models are first and foremost computational engines, and that their full potential will be realized not by better prompting but by better programming models. Chat is dominant mode of interaction with language models today, and anthropormophism pervades nearly every aspect of our usage of models. We argue that by understanding models instead as compute elements with unique memory and compute mechanism, we can make agents (and other programs built atop them) more predictable, reliable, and secure.

This document describes our vision for an end-to-end implementation of generative computing ideas. The document spans from a formal algebra of composable KV cache regions, through a token-level markup language that enforces instruction/data separation, to a developer library for writing reliable generative programs, to a semantic layer where systems reason and converse over typed meanings, to a deployment model where composable pipelines present as standard model endpoints.

**Mellea** is the open-source library that reduces generative computing to practice: `mellea-core` (currently in Python, and only partially completed) provides the infrastructure; `mellea.stdlib` (Python) provides the developer-facing patterns.

The full promise of generative computing will be unlocked when models and software can be codesigned. The primary target for our codesign efforts is **Granite** — IBM's own model family, whose architecture and training we control. We have been steering model architectures in directions to support this codesign since Granite 4.0.


---

## Table of Contents

### [Chapter 0: Introduction](chapters/00-introduction.md)
> The generative computing proposition. Granite as primary target architecture. Mellea as reduction to practice. The abstraction boundary: from full structural enforcement to prompt-assembly approximation. Three-part document structure.

---

### Part I: The Generative Computing Compute Model

*Theoretical foundations and `mellea-core` infrastructure. What is the computational substrate for generative computing?*

**[Chapter 1: The Span Algebra](chapters/01-span-algebra.md)**
> Spans, KV cache representations, prefill/generation, scope algebra (union and composition), fan-out and fan-in, tainting vs. positional entanglement, the free monoid under NoPE, tagging, provenance, and taint breaking via refill.

**[Chapter 2: LLMON](chapters/02-llmon.md)**
> Why LLMs need their own data representation. Syntax collisions, injection, special tokens as the solution. Six design requirements. Human-friendly and Machine LLMON syntaxes (six special tokens). Expressivity: referential binding, exec mechanism, prompt injection defense. Boundary-constrained masking. LLMON in training. Empirical results (~83% distractor robustness with training, ~72% with inference-time masking alone).

**[Chapter 3: Model-Runtime Codesign](chapters/03-model-runtime-codesign.md)**
> The codesign principle: models and runtimes designed together. `mellea-core` as the runtime implementation. The abstraction boundary: native enforcement vs. prompt-assembly approximation. The three-layer architecture. Integration surfaces: KV pool as Level 0, scope resolution at two scales. Algebra-to-runtime primitive mapping. What training owes serving.

**[Chapter 4: Cache-Materialized Training](chapters/04-cache-materialized-training.md)**
> Closing the train/serve gap: span-sequential forward pass with explicit KV cache materialization at span boundaries. Gradient flow through cache writes. Two-phase curriculum. Span parallelism via dependency DAGs. Feasibility analysis (FLOP parity, ~1.05–1.15× overhead). Optional learned transforms (compression, eviction, merging). Implementation on LM Engine.

**[Chapter 5: Serving Engine](chapters/05-serving-engine.md)**
> NoPE-native serving: KV pool with slot allocator, span registry with provenance, gather-based attention, cascade attention for shared-scope batching, scope-isolated masking, refill (taint breaking). Request lifecycle, scheduler policy, scope-aware admission. API design (fill, gen, gen_parallel, compose, refill, execute).

**[Chapter 6: Span Attention](chapters/06-span-attention.md)**
> Hierarchical coarse-to-fine gating for corpus-scale contexts (10⁶–10¹² tokens). SpanSummary trait (vMF/ScaledSphere metrics), centroid fitting, potency. IndexTree (multilevel centroid hierarchy). Stick-breaking attention for budget-constrained tree descent. Tiered storage and eviction (GPU → host → SSD → object store). Offline corpus precomputation. FA-3 integration.

---

### Part II: High-Level Generative Computing Patterns

*`mellea.stdlib` and the semantic layer. How do developers write generative programs?*

**[Chapter 7: Mellea Standard Library](chapters/07-mellea-stdlib.md)**
> The encapsulation principle and LLM intrinsics. MelleaSession. Instructions and requirements (IVR: instruct-validate-repair). Generative slots (`@generative`). MObjects (`@mify`): the bridge between traditional and generative computing (mirroring, manipulation, creation, pass-by-value/reference). CBlocks and context management. Backend-agnostic design. Circumscribing nondeterminism.

**[Chapter 8: Meaning and Thought](chapters/08-meaning-and-thought.md)**
> Two primitives: Meaning (semantic content with lazy transforms — negate, paraphrase, abstract, concretize, reframe, compress, elaborate) and Thought (typed cognitive operation over Meanings with IVR). Types as Meanings (soft semantic type system). Thought execution, validation, selection, traces, composition (`then`, `gate`), injection points, failure and degradation.

**[Chapter 9: Decision Architecture](chapters/09-decision-architecture.md)** *(placeholder)*
> Decision points, signals, and policies for Thought selection. Grammar-based constraints, pragmatic overrides, pluggable optimizers. Decision logging and three-way attribution.

**[Chapter 10: Conversation](chapters/10-conversation.md)**
> NCF (Natural Conversation Framework) as a Thought grammar. Intake pipeline: raw text → typed Meanings at a hard security boundary. Slot filling with Meaning-based elicitation. Process flows (generator functions with abort semantics). Developer experience. Safety model: user text never appears in an instruction position.

**[Chapter 11: Reasoning](chapters/11-reasoning.md)**
> Seven-phase core pipeline (characterize → extract_info → decompose → ground → execute → format_plan → synthesize). 47-pattern Thought catalog (content-transforming, structure-changing, evaluation, robustness, meta). ThoughtSequence profiles. Pluggable optimizer (UCB bandit, contextual bandit, LLM-as-selector). Meanings as data, not instructions (preventing self-anchoring and reasoning contamination).

**[Chapter 12: Unification, Traces, and Synthetic Data](chapters/12-unification-traces-sdg.md)**
> Conversation and reasoning as one mechanism (same Meaning operations, different verbalization). Cross-catalog invocation. Four invocation modes (prompted → LLMON external → LLMON internal → mixed control). LLMON trace encoding. Synthetic data generation: reasoning traces, conversation traces, bridge traces, systematic variation via Meaning operations, quality controls.

**[Chapter 13: Semantic Operators](chapters/13-semantic-operators.md)** *(placeholder)*
> The formal operator algebra over Meanings. Composition laws, operator-level type constraints, and the connection to LLMON's semantic tagging system.

---

### Part III: Generative Computing Systems

*Deployment, security, and application-level patterns. How does generative computing meet the real world?*

**[Chapter 14: Software-Defined Models](chapters/14-software-defined-models.md)**
> SDMs: composing model behavior from weights, programs, validators, and control flow behind a standard OpenAI-compatible endpoint. SDMs vs. direct Mellea programming (two modes, not alternatives). The simplicity spectrum (passthrough → observation → remediation → smart pipeline → full composition). Why SDMs matter most for small models. Connection to the generative computing stack. The SDM as a versionable artifact. Capabilities as plugins.

**[Chapter 15: Security Across the Stack](chapters/15-security.md)**
> *(Placeholder — requires expansion with formal threat models.)* Taint analysis via spans and scopes. Instruction/data separation via LLMON. Policy enforcement via the runtime. Untrusted input and the semantic boundary (intake pipeline, Meaning/Thought data-plane isolation). Defense in depth across all layers. Open questions.

**[Chapter 16: Multi-Agent Patterns](chapters/16-multi-agent-patterns.md)**
> Agent isolation via scopes and taint tracking. Inter-agent communication via typed Meanings. Orchestration patterns (orchestrator-worker, evaluator-optimizer, role-based teams, hierarchical delegation) mapped to span algebra operations. State and memory in multi-agent systems. Security at agent boundaries.

---

## How to Read This

| You are a... | Start here |
|---|---|
| **Model trainer** | [Ch. 1](chapters/01-span-algebra.md) → [Ch. 2 §2.6–2.7](chapters/02-llmon.md) → [Ch. 4](chapters/04-cache-materialized-training.md) |
| **Serving engineer** | [Ch. 1](chapters/01-span-algebra.md) → [Ch. 5](chapters/05-serving-engine.md) → [Ch. 3 §3.5–3.6](chapters/03-model-runtime-codesign.md) |
| **Application developer** | [Ch. 7](chapters/07-mellea-stdlib.md) → [Ch. 8](chapters/08-meaning-and-thought.md) → [Ch. 10](chapters/10-conversation.md) or [Ch. 11](chapters/11-reasoning.md) |
| **RAG / retrieval researcher** | [Ch. 1](chapters/01-span-algebra.md) → [Ch. 6](chapters/06-span-attention.md) → [Ch. 3 §3.5](chapters/03-model-runtime-codesign.md) |
| **Multi-agent builder** | [Ch. 8](chapters/08-meaning-and-thought.md) → [Ch. 16](chapters/16-multi-agent-patterns.md) → [Ch. 7 §7.10](chapters/07-mellea-stdlib.md) |
| **Hardware / infra engineer** | [Ch. 1](chapters/01-span-algebra.md) → [Ch. 5](chapters/05-serving-engine.md) |
| **System architect** | All chapters in order |
