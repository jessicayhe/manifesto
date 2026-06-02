> [← Ch.13: Semantic Operators](13-semantic-operators.md) · [Table of Contents](../README.md) · [Ch.15: Security Across the Stack →](15-security.md)

---

# Chapter 14: Software-Defined Models

**Status:** Draft
**Version:** 0.5
**Authors:** David Cox, Claude
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-06

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## Composing Model Behavior from Weights, Programs, Validators, and Control Flow

When a client calls an OpenAI-compatible endpoint, it thinks it's talking to a model. This chapter proposes that it should be — but the model's behavior need not be defined solely by its weights.

A **software-defined model (SDM)** composes weights, input transforms, output transforms, validators, control flow, and programmatic logic into a single entity that is indistinguishable from a conventional model at the API surface. Every component of this composition is pluggable and optional. An empty SDM — the default — is a pure passthrough, identical in behavior to a conventional endpoint. Each capability is added incrementally by plugging in a transform or swapping the controller. The SDM is defined by a declarative configuration file — a versionable, testable artifact — and executed by Mellea.

The analogy to **software-defined networking (SDN)** is precise and intentional. SDN decoupled network behavior from physical switches and routers, making routing decisions programmable through a software abstraction layer. An SDM decouples model behavior from the weights, making generation decisions programmable through a pipeline abstraction layer. The weights are still there — they do the generation work — but the behavior of the model is defined by software.

## 12.0 SDMs and Direct Mellea Programming: Two Modes of Interaction

The generative computing stack supports two modes of interaction, and the SDM is only one of them.

**Direct Mellea programming** is the primary mode. A developer writes a generative program using `mellea.stdlib` (Chapter 7) — instructions with requirements, generative slots, MObjects, Meaning/Thought pipelines — and runs it against any backend. The developer controls the full expressiveness of the stack: span composition, scope management, IVR loops, Thought selection, conversation flows. This is how new applications are built from the ground up. The preceding chapters (7–11) describe this mode.

**Software-defined models** are the deployment mode. An SDM wraps the generative computing stack behind a standard OpenAI-compatible endpoint, making it invisible to existing tools and frameworks. The client sends a prompt; the SDM's pipeline — which is itself composed of Mellea programs — processes it through transforms, a controller, and validators, and returns a response. The client never knows that the "model" is a composition of weights and software. This is how generative computing meets existing ecosystems where they are — behind the protocol they already speak.

The two modes are not alternatives. They are layers. An SDM's transforms and controllers *are* Mellea programs. A guardrails transform uses `m.instruct()` with requirements. A reasoning controller invokes the Meaning/Thought pipeline (Chapter 11). A tool-selection transform uses `@generative` functions. The SDM is a deployment wrapper around the same building blocks that a developer uses directly. The difference is who is in control: in direct programming, the developer writes the program; in an SDM, the pipeline definition specifies which programs run and in what order, and the client's request triggers execution.

This means a team can start in either mode. A team building a new product writes Mellea programs directly. A team integrating Granite into an existing LangChain pipeline deploys an SDM that transparently upgrades the pipeline's capabilities. And as the team matures, they can move from SDM-mediated interaction to direct Mellea programming for components that need more control — or the other way around, wrapping a mature Mellea program in an SDM to expose it as a model endpoint for other teams.

## 12.1 How It Works

The SDM framework exposes a standard OpenAI-compatible endpoint — a virtual inference endpoint. Behind it, three stages execute in sequence. Each is pluggable and optional:

**Input transforms (optional)** intercept the request — diagnosing prompt pathologies, rewriting for model-specific formatting, pruning tool lists, enhancing RAG context, enforcing guardrails. Each transform can make its own LLM calls via Mellea. If no input transforms are configured, the request passes through unchanged.

**A controller (pluggable)** performs generation — through Mellea's instruct-validate-repair loop (Chapter 7, §7.3), through inference scaling, through a Meaning/Thought reasoning pipeline (Chapter 11), or through any other pluggable strategy. The default controller is a simple passthrough that forwards the request to the underlying model endpoint and returns the response unmodified.

**Output transforms (optional)** intercept the response — scoring uncertainty, enforcing policies, repairing violations, attaching metadata. If no output transforms are configured, the response passes through unchanged.

The entire pipeline is defined by a declarative configuration file. This file is the SDM definition. Two SDMs can share the same base weights but behave completely differently because their definitions differ.

Mellea is the execution backbone: every stage of the pipeline — not just the controller — has access to Mellea's LLM dispatch, instruct-validate-repair loops, intrinsic functions (Chapter 7, §7.1), and programmatic patterns. This is what makes transforms first-class programs rather than simple rewrite rules.

## 12.2 The Simplicity Spectrum

A critical property: the SDM framework imposes no minimum complexity. The simplest possible SDM is an empty definition — no transforms, default passthrough controller — which behaves identically to a conventional model endpoint with zero overhead. Every capability is added by plugging in a component, and every component can be removed without breaking anything else.

SDMs exist on a spectrum:

**Bare passthrough.** No plugins configured. The request goes in, the response comes out, nothing is modified. The starting point for any deployment, and the default.

**Observation only.** A single input transform that logs diagnostics — prompt quality scores, formatting issues, tool count warnings — without modifying anything. The simplest useful SDM: deploy it, learn what your traffic looks like, change nothing.

**Light remediation.** A few input transforms that fix known issues — reformat tool definitions, inject a system prompt, truncate long conversations. No LLM calls in the pipeline, just deterministic rewrites.

**Smart pipeline.** Input and output transforms that make their own LLM calls — guardrails via Granite Guardian, tool selection via pairwise comparison, uncertainty quantification. The default controller forwards to the model; the transforms do the heavy lifting.

**Controller-centric.** No transforms at all. A sophisticated Mellea controller implements the full generation strategy — instruct-validate-repair, inference scaling, Meaning/Thought reasoning (Chapter 11), or a custom Mellea program. All the intelligence is in the controller.

**Full composition.** Input transforms, a Mellea-backed controller, and output transforms, all working together. The most powerful configuration but also the most complex — and it is never required.

The point is not that every SDM should use every capability. It is that any capability can be added to any SDM at any time, without modifying the rest of the pipeline, and without the client knowing or caring.

## 12.3 Why This Matters Most for Small Models

The SDM concept is useful at any scale, but its value is disproportionately high for small language models (SLMs). Large frontier models absorb messy prompts, noisy tool lists, and unstructured RAG context through sheer capacity — SLMs cannot. Every wasted token, every formatting inconsistency, every ambiguous tool description costs a small model more than it costs a large one.

An SDM turns a small model's sensitivity into a design opportunity. The software layer compensates precisely where the weights fall short — at a fraction of the cost and latency of scaling up the weights.

For the Granite ecosystem specifically: a Granite 3B model inside an SDM can match or exceed the effective capability of a much larger model, at a fraction of the cost and latency, and with stronger guarantees. The SDM is what makes the small model viable for production use cases that would otherwise demand a frontier model.

Three mechanisms drive this:

**Prompt hygiene disproportionately benefits SLMs.** Large models are robust to prompt format inconsistencies. SLMs are not. The same prompt that works fine with a 70B model may fail silently with a 3B model. The SDM detects model-specific format violations and rewrites the request into the form the target SLM handles best — especially high-leverage when the SLM is accessed through a framework (LangChain, LlamaIndex) designed and tested against large models.

**Inference scaling, IVR, and divide-and-conquer.** Multiple cheap SLM calls with programmatic control flow can substitute for a single expensive frontier-model call. Inference scaling runs the same query N times and selects by consensus. IVR catches errors and gives the model a focused repair prompt — small models are often good at targeted fixes even when first-pass generation is imperfect. Divide-and-conquer decomposes hard problems (select from 20 tools) into many easy problems (compare 2 tools) that SLMs handle well.

**Dynamic adapter selection.** An input transform inspects the request, classifies the task, and routes to the correct LoRA-equipped endpoint. This is input-dependent capacity allocation — the same base SLM dynamically leans toward the specific capability the request demands. The client doesn't think of itself as talking to "a 3B model plus an adapter router." It's talking to a model — one whose capabilities happen to be software-defined.

These mechanisms reinforce each other. A request might arrive with a messy prompt and 25 tool definitions, targeting a 3B Granite model. Prompt hygiene cleans the format. Tool selection prunes to the 4 most relevant via pairwise comparison. Adapter selection routes to the tool-calling LoRA. The controller executes via IVR. Guardrails validate the output. Each step is independently valuable. Together, they enable a small, cheap, fast model to produce results that would otherwise require a model 10–20× its size.

## 12.4 Connection to the Generative Computing Stack

The SDM is where the generative computing infrastructure meets the outside world. Every layer of the stack contributes:

**Span algebra (Chapter 1).** The SDM's transforms operate on spans. An input transform that restructures the prompt is performing span operations — retagging, rescoping, composing. Under NoPE (Granite), these operations are free-monoid compositions: zero compute, position-invariant.

**LLMON (Chapter 2).** When the model is LLMON-aware, transforms can emit and consume LLMON structure. An input transform that tags instruction and data regions produces LLMON spans; the controller's boundary-constrained masking enforces the boundaries structurally. The SDM is the layer that introduces LLMON structure when the client doesn't provide it.

**Model-runtime codesign (Chapter 3).** The SDM is the practical realization of the abstraction boundary (Chapter 3, §3.3). The client sees a model. Behind the API surface, the SDM implements the native/approximation split: a full `mellea-core` runtime with Granite for structural enforcement, or prompt-assembly approximation for any backend. The client neither knows nor cares which implementation is active.

**`mellea-core` (Chapters 4–6).** The controller can use the `mellea-core` serving engine directly — managing KV pools, span registries, and scope-isolated masking at the infrastructure level. For corpus-scale contexts, the span attention index (Chapter 6) selects the working set; the SDM's KV pool materializes it.

**`mellea.stdlib` (Chapter 7).** Transforms and controllers are `mellea.stdlib` programs. They use `m.instruct()`, `@generative` functions, MObjects, and sampling strategies. The encapsulation principle (Chapter 7, §7.1) applies: a transform might start as a prompted instruction and later be replaced by an activated LoRA intrinsic — the SDM definition doesn't change.

**Meaning/Thought (Chapters 8–11).** A controller-centric SDM can implement a full Meaning/Thought reasoning pipeline (Chapter 11) or an NCF-governed conversation (Chapter 10). The client sends a prompt; the SDM's controller decomposes it into Thoughts, executes them, and returns the synthesized answer. The client sees a model that thinks deeply — but the thinking is software-defined.

## 12.5 The SDM as Artifact

The SDM definition is a declarative configuration file — the model's identity in a versionable, testable, reviewable form:

**Versionable.** The SDM definition lives in version control alongside application code. Changes to model behavior are tracked as diffs.

**Testable.** An SDM can be tested with a suite of input/output pairs. Regressions in model behavior are caught by CI, not by users.

**Reviewable.** Changes to an SDM definition go through code review. Adding a guardrail, changing a policy rubric, or adjusting tool selection thresholds are all visible, discussable decisions.

**Composable.** Transforms are independently developed and shared. A team publishes a guardrails transform as a reusable package; another team composes it into their SDM definition.

**Promotable.** The same SDM definition runs in dev, staging, and production. Promoting a model to production means moving a file.

The separation between the SDM definition (what the model is) and the server configuration (how it is deployed) reinforces this. The same SDM in debug mode and production mode may differ only in their server config — the model identity is the definition.

## 12.6 Capabilities as Plugins

The following are optional plugins for the SDM framework. Each is independently available as a transform or controller. No capability depends on any other being present, and an SDM with none of them configured is a valid passthrough.

**Debug: transparent diagnostics.** Pass traffic through unaltered, inspect for pathological patterns, log structured diagnostics. Rule-based checks catch formatting issues; LLM-based checks assess system prompt quality and RAG relevance. Pure observation mode.

**Remediation: proactive input repair.** Same diagnostics, but the SDM actively rewrites the request to fix detected issues before forwarding. Tool definitions moved to the right field, missing descriptions generated, RAG documents repositioned, system prompt rewritten against a quality rubric. The client sees none of this — just better results.

**RAG enhancement via intrinsic functions.** Detect RAG patterns and invoke Mellea intrinsics: relevance filtering, passage reranking, query decomposition, citation scaffolding, faithfulness priming. A naïve RAG pipeline is transparently upgraded to a sophisticated one.

**Guardrails via Granite Guardian.** Safety, quality, and policy checks on both input and output. Input: prompt injection, jailbreak attempts, PII detection, toxicity screening. Output: hallucination detection, PII leakage, factual consistency, policy compliance. Each detection is configurable: log, block, redact, or repair.

**Tool selection for SLMs.** When presented with more tools than the SLM can reliably select from, the SDM decomposes tool selection into structured multi-round comparison. Pairwise comparison turns a hard problem into many easy problems.

**Uncertainty quantification.** Calibrated confidence score for every response, attached as a response field and HTTP header. Responses below a threshold can trigger re-generation or be flagged for human review.

**Policy enforcement.** Configurable policy checks on every response — code-based (length, format, required sections) and LLM-based (tone, accuracy, completeness). On violation, targeted repair via IVR loop. Original and repaired versions logged for audit.

**Automatic inference scaling.** Run the same query N times in parallel with varied temperature, select by consensus. Strategies: majority vote, best-of-N, merge. Especially powerful for SLMs where any single call may be unreliable but the ensemble is robust.

**Prompt-to-program (conceptual).** Automatically decompose a complex prompt into a Mellea program — a structured DAG of LLM calls with programmatic control flow. The SDM analyzes the prompt, identifies subtasks and dependencies, generates focused prompts for each, and executes the program. The result is packaged as a single response.

## 12.7 Open Questions

**Streaming semantics.** Many output transforms need the full response. In streaming mode, should the pipeline buffer before running output transforms, or allow transforms to process chunks?

**Cost and latency budgets.** Transforms that make their own LLM calls add latency. Should the pipeline enforce a budget — a maximum number of calls per request, or a latency ceiling?

**Per-model transform profiles.** Different models need different remediation, tool thresholds, and guardrails. Should the SDM definition support per-model overrides within a single definition?

**Observability.** Beyond diagnostic logs, should the SDM framework expose Prometheus metrics, OpenTelemetry traces, or a real-time dashboard for each request's pipeline journey?

**Composition of SDMs.** Can an SDM's backend be another SDM? This would enable hierarchical composition — a routing SDM that dispatches to specialist SDMs, each with their own transforms. The OpenAI-compatible interface makes this structurally possible.

---

> [← Ch.13: Semantic Operators](13-semantic-operators.md) · [Table of Contents](../README.md) · [Ch.15: Security Across the Stack →](15-security.md)
