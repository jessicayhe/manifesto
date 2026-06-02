> [← Ch.14: Software-Defined Models](14-software-defined-models.md) · [Table of Contents](../README.md) · [Ch.16: Multi-Agent Patterns →](16-multi-agent-patterns.md)

---

# Chapter 15: Security Across the Stack

**Status:** Placeholder draft — extracted from the Generative Computing technical memo (March 2026). This chapter collects the security analysis that is currently distributed across the preceding chapters into a unified cross-cutting treatment. It requires expansion with formal threat models, concrete attack/defense scenarios, and empirical validation beyond the LLMON distractor results reported in Chapter 2. The structure is sound; the content needs depth.

**Version:** 0.1
**Authors:** David Cox, Claude, Ambrish Rawat
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-14

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## Security as a Structural Property

A recurring theme throughout this document is that the generative computing framework provides **structural** security properties — guarantees that follow from how the system is built, rather than from hoping that the model behaves well. This chapter traces those guarantees across every layer of the stack, from the KV cache through the runtime to the semantic layer.

The key principle: security should be a property of the **architecture**, not a behavior learned by the **model**. The same principle that makes operating systems distinguish code segments from data segments, and for the same reason: the alternative (allowing data to be executed) is the root cause of an entire class of vulnerabilities.

## 15.1 Security Questions

The preceding chapters argue that generative computing should be treated as a programming model, not as a prompting technique. The same claim has a security consequence: if the system is a programming model, then its security properties should be stated in terms of structure, authority, data flow, and enforcement boundaries, not in terms of whether the model "usually behaves well."

The core security questions of the stack are therefore simple:

**Information-flow security:** *what can influence what?*

**Authority security:** *what can act on behalf of whom?*

These two questions organize the rest of the chapter. The span algebra addresses information-flow at the KV level. LLMON addresses authority at the token level. The runtime mediates both. The Meaning/Thought layer carries the same separation upward to the semantic boundary.

## 15.2 Threat Model

We assume an attacker who can control or influence content-bearing channels that enter the system as data. This includes direct user input, retrieved documents, tool returns, object fields exposed to the model, conversation history imported from outside the current trust domain, and model-produced intermediate text from earlier steps. The attacker's goals may include causing unauthorized influence on downstream computation, exfiltrating sensitive information, inducing policy violations, corrupting reasoning, or causing content that should remain data to acquire the force of authority.

We do **not** assume that the base model is trustworthy in the strong sense. The model may be distractible, may overgeneralize, may follow adversarial patterns if allowed to see them in an instruction-like role, and may imperfectly respect conventions. The purpose of the architecture is precisely to reduce reliance on such behavioral assumptions.

We also do not assume that all components outside the runtime are trustworthy. Retrieved corpora may be poisoned. Tool outputs may contain adversarial strings. Application objects may contain secrets adjacent to benign fields. Intermediate model outputs may contain falsehoods or contaminated summaries. The stack should therefore be analyzed under the assumption that untrusted content is common, not exceptional.

### 15.2.1 Adversary

The adversary in this chapter is primarily a **data-plane adversary**: an entity that can supply content to the system through channels that are intended to carry data rather than authority. In the simplest case this is an end user. In more realistic deployments it also includes untrusted document sources, tool providers, imported transcripts, application objects whose fields are surfaced to the model, and prior model outputs that re-enter the system as context.

This section should eventually distinguish among attacker classes more precisely — for example, direct user attackers, retrieved-content attackers, tool-channel attackers, and cross-session contamination sources — but for now the common case is sufficient: the attacker can control inputs that the system is supposed to treat as data and will attempt to exploit any ambiguity by which data might become control.

### 15.2.2 Attacker Control Surface

The attacker may control or influence:

- raw user text
- retrieved passages and corpus content
- tool returns and API responses
- object fields or serialized application state exposed to the model
- imported conversation history
- intermediate textual artifacts from earlier model calls

The attacker is **not** assumed, in this chapter, to control the runtime's policy code, the structural tokens of LLMON, or the implementation of scope resolution itself. Those belong to the trusted side of the architecture. A fuller treatment should later distinguish partial compromise of these components from the baseline attacker considered here.

### 15.2.3 Trusted Computing Base

The minimal trusted computing base for the structural claims in this chapter includes:

- the runtime machinery that constructs spans, resolves scopes, and applies enforcement
- the LLMON parser and boundary machinery
- the provenance and taint-tracking metadata maintained by the system
- the intake pipeline that converts raw text into typed Meanings
- the policy and validation logic that mediates tool use and output release

These claims do **not** require the model to be trusted in the same way. The model is treated as a powerful but imperfect computational component operating inside a structured environment. This is the key shift in trust placement: trust is anchored in the architecture and its mediation points, not in the model's behavioral reliability.

A **control-plane adversary** is possible in principle: an attacker who can modify runtime policy, structural bindings, parser behavior, or other trusted enforcement machinery. Such an attacker lies largely outside the baseline model of this chapter, because it targets the trusted computing base itself. The chapter's structural guarantees therefore depend on the integrity of these control-plane components.

### 15.2.4 Structural vs. Statistical Guarantees

The chapter distinguishes between **structural** guarantees and **statistical** ones.

A structural guarantee is one enforced by the architecture itself: scope exclusion, explicit instruction binding, special-token boundary separation, runtime mediation, and semantic data-plane isolation. These guarantees do not depend on whether the model has merely learned a convention.

A statistical guarantee is one that depends on the model's learned behavior: recognizing intent, resisting adversarial phrasing by habit, preserving distinctions that are present only as soft prompting conventions, or complying with policies because similar behavior appeared in training data.

Much of the motivation for generative computing is to move critical security properties from the second category into the first. This subsection should later be expanded with a more explicit inventory of which guarantees in each deployment mode are structural, which remain statistical, and which are hybrids.

### 15.2.5 Security Boundaries

The threat model is organized around a small number of explicit security boundaries.

- At the **token boundary**, structural delimiters are separated from ordinary text, so that syntax and content do not occupy the same channel.
- At the **scope boundary**, visibility and dependency are made explicit through spans and scopes, so that information-flow is represented structurally rather than inferred from a flat prompt.
- At the **runtime boundary**, execution, policy enforcement, and tool access are mediated rather than delegated to model behavior.
- At the **semantic boundary**, raw text is consumed into typed Meanings so that downstream computation operates over semantic data rather than ambient text.

These boundaries are not all equally strong in every deployment mode. The strongest claims in this chapter apply where the runtime and model cooperate to enforce them natively; weaker approximations preserve some of the abstractions but not all of the guarantees.

## 15.3 Security Invariants

The security claims of the stack are organized around two invariants.

First, **information-flow security**: influence should follow explicit structure. If one span, input, or intermediate artifact is allowed to affect another computation, that influence should arise through a defined scope, dependency, or provenance relation, not through accidental entanglement in an unstructured prompt.

Second, **authority security**: authority should be granted explicitly, not inferred from surface form. Content does not become executable, instruction-bearing, or tool-invoking merely because it appears imperative or occurs in a privileged textual position. Authority must be assigned through explicit bindings and runtime mediation.

These are structural claims. They do not require the model to be perfectly reliable, and they do not eliminate every possible failure mode. They instead reduce the security problem to explicit control of influence and explicit assignment of authority.

## 15.4 Taint Analysis via Spans and Scopes

In conventional LLM inference, every token attends to every preceding token. Sensitive data introduced at any point — PII, confidential business information, implementation secrets — taints every subsequent KV cache entry. There is no post hoc mechanism to undo this tainting; the information has been irreversibly mixed into the model's internal representations.

The span algebra (Chapter 1) makes taint analysis tractable. Because scopes are explicit and the dependency algebra is well-defined (Chapter 1, §1.3), the runtime can track precisely which spans have been influenced by which other spans. If a span is tagged as SENSITIVE, the runtime knows that every span whose scope includes the sensitive span is tainted — and can compute alternative generation paths that avoid the tainted scope, producing clean outputs when needed.

This is air-tight taint analysis: it works even in the presence of the KV cache's inherent tendency to propagate information downstream, because the analysis operates at the scope level rather than the token level. The scoping machinery makes explicit what was previously an invisible and uncontrollable side effect.

The **refill** operation (Chapter 5, §10) is the active mechanism for breaking taint: discard a span's KV, recompute it under a clean scope, update the provenance record. Under Granite's NoPE attention layers, refill produces position-invariant KV that is valid at any slot position — there is no positional side effect from the recomputation.

## 15.5 Instruction/Data Separation via LLMON

Prompt injection — the insertion of adversarial instructions into content that the model should treat as data — is the defining security challenge of LLM-based systems. The conventional defense relies on the model learning, through training, not to follow instructions embedded in data regions. A statistical defense: it works most of the time, but a sufficiently clever adversary can craft inputs that exploit the model's inability to make a hard distinction between instructions and data.

LLMON's exec mechanism (Chapter 2, §2.4) provides a structural alternative. When instructions and data are explicitly tagged as separate spans, and execution is bound to a specific named instruction via an exec span, the system no longer relies on the model to determine what should be executed. The runtime knows — from the markup — which span is the instruction, which spans are data, and which instruction has been selected for execution.

Boundary-constrained masking (Chapter 2, §2.5) enforces this structurally at the attention level: distractor instructions and adversarial content in data spans are masked out of the attention computation entirely, not merely hoped to be ignored. The empirical results reported in Chapter 2 (§2.7) demonstrate that this approach raises distractor robustness from near-zero to above 83% with structured training, and from ~41% to ~72% with inference-time masking alone.

LLMON's use of special tokens (Chapter 2, §2.1) raises the bar further. Because structural delimiters exist outside the expressive range of regular text, a user typing ordinary input cannot produce them. The tokenizer will never segment ordinary text into a sequence that includes structural tokens. Injection via syntax collision — the mechanism behind both SQL injection and prompt injection — is eliminated at the token level.

## 15.6 Policy Enforcement via the Runtime

The runtime (Chapter 3) sits between the developer's program and the model, mediating every interaction. This mediation point creates a natural location for policy enforcement: the runtime can inspect, validate, and modify spans before they reach the model and after the model produces output, applying organizational policies that are orthogonal to what the model was trained to do.

`mellea.stdlib` (Chapter 7) provides the infrastructure for this through several mechanisms. Requirements in the instruct-validate-repair loop (Chapter 7, §7.3) can encode safety policies as validation checks — the runtime checks that outputs conform to organizational constraints before they reach the user. The Granite Guardian library provides pre-built safety, factuality, and policy compliance verifiers that can be attached to any generative call. The MObject pattern (Chapter 7, §7.5) controls which fields and methods of an object are exposed to the model, limiting the attack surface.

More broadly, the runtime's role in structuring content is itself a security property. In conventional LLM usage, the prompt is an unstructured string assembled by the developer, and the model receives it as-is. In generative computing, the runtime assembles the scope, applies tags, manages the KV cache, and constructs the token stream from typed components. The runtime controls what the model sees and how it sees it. The model operates within a structured environment, not on a raw string — much as a CPU executes instructions within the constraints of an operating system, not having unrestricted access to memory and I/O.

## 15.7 Untrusted Input and the Semantic Boundary

The Meaning/Thought framework (Chapters 8–11) takes the security story to its logical conclusion. In the conventional model, user input is a string that is concatenated into a prompt, and any instruction-like content in that string may influence the model's behavior. In the Meaning/Thought model, user input passes through an **intake pipeline** (Chapter 10, §3.2) that parses it into typed Meaning objects at a hard security boundary. The raw text is consumed by the parser and does not propagate further into the system.

After the intake boundary, there is no text in the system that could be confused with instructions — only Meanings, which are typed semantic objects that flow through typed cognitive operations (Thoughts). A Thought's instruction is the control plane; input Meanings appear only in the data plane (`user_variables`). Even if a user's input contains adversarial instruction-like content, that content is captured as a Meaning — a piece of semantic data — and can never reach an instruction position.

Similarly, the system's own intermediate results (produced by reasoning Thoughts) are Meanings, not authoritative text. Downstream Thoughts use them as material to be manipulated, not as instructions to follow. This prevents **self-anchoring** — the model treating its own intermediate output as ground truth — and **reasoning contamination**, where a flawed intermediate result corrupts subsequent steps.

The enforcement model is layered across the four invocation modes (Chapter 12, §5.4):

**Mode 0 (prompted).** Data isolation is enforced by Mellea's API: Meanings appear in `user_variables` (data plane), never in `description` (control plane). This is a strong convention — the same control/data plane separation used by the security community — but it is not a structural guarantee at the token level.

**Modes 1–3 (LLMON single-stream).** LLMON constraint masking provides structural enforcement: data spans are prevented from being attended to as instructions at the attention level. The convention becomes a guarantee.

## 15.8 Defense in Depth

The security properties of the generative computing stack are cumulative. Each layer addresses a different aspect of the problem, and the layers compose:

| Layer | Mechanism | Threat Addressed |
|---|---|---|
| Spans and Scopes (Ch. 1) | Taint tracking, scope isolation | Sensitive data leakage |
| LLMON (Ch. 2) | Instruction/data separation, boundary masking, special-token delimiters | Prompt injection, syntax collision |
| Runtime (Ch. 3, 7) | Policy enforcement, structured content assembly, attack surface control | Organizational policy violation |
| `mellea.stdlib` (Ch. 7) | Requirements validation, controlled field exposure | Output quality, attack surface |
| Meaning/Thought (Ch. 8–11) | Intake parsing, semantic typing, data-plane isolation | Untrusted input, self-anchoring, reasoning contamination |
| SDMs (Ch. 12) | Guardrails, input remediation, output policy enforcement | Production-grade defense composition |

No single layer is sufficient. A model trained on LLMON structure might still be fooled by a sufficiently adversarial input — but the runtime's boundary-constrained masking catches what the model misses. A policy verifier might miss a subtle violation — but the intake pipeline has already stripped the raw text and prevented instruction injection at the source. Each layer provides guarantees that do not depend on the layers above or below.

The layers compose because they control influence and authority at different levels: scope and provenance at the KV layer, instruction binding at the token layer, policy mediation at the runtime layer, and data-plane isolation at the semantic layer.

## 15.9 Open Questions

**Formal threat modeling.** This chapter traces structural guarantees but does not define a formal threat model with attacker capabilities, attack surfaces, and security boundaries. A rigorous treatment should enumerate the attack vectors that each layer addresses and characterize the residual attack surface.

**Special-token spoofing.** LLMON's special tokens raise the bar for injection but do not eliminate it entirely. Advanced adversarial techniques could attempt to craft inputs that produce activation patterns similar to special tokens. The feasibility and mitigation of this attack class needs empirical characterization.

**Taint propagation through learned transforms.** If the optional learned cache transforms (Chapter 4, §5.3 — compression, eviction, merging) are enabled, do they preserve or degrade taint-tracking guarantees? A compressed cache entry derived from a SENSITIVE span should itself be considered tainted, but the compression may mix tainted and untainted content in ways that complicate tracking.

**Cross-model taint.** If multiple models share a KV cache or span index (Chapter 6, Open Question 11), taint from one model's processing could propagate to another model's context. The provenance model needs extension to handle multi-model scenarios.

**Empirical validation at scale.** The distractor robustness results (Chapter 2, §2.7) are preliminary, covering two 3B-parameter models on one benchmark. Comprehensive evaluation across model sizes, attack sophistication levels, and real-world deployment scenarios is needed to validate the defense-in-depth claims.

---

> [← Ch.14: Software-Defined Models](14-software-defined-models.md) · [Table of Contents](../README.md) · [Ch.16: Multi-Agent Patterns →](16-multi-agent-patterns.md)
