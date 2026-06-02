> [← Ch.10: Conversation](10-conversation.md) · [Table of Contents](../README.md) · [Ch.12: Unification, Traces, and Synthetic Data →](12-unification-traces-sdg.md)

---

# Chapter 12: Reasoning

**Component:** `mellea.stdlib`
**Status:** Draft
**Version:** 1.1
**Authors:** David Cox, Claude
**Date Created:** 2026-03-24
**Date Modified:** 2026-04-06

**Prerequisites:** Chapter 8 (Meaning and Thought primitives)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## Structured Reasoning as Thought Sequences

This chapter applies the Meaning/Thought framework (Chapter 8) to structured reasoning: decomposing complex thinking into many small, focused LLM calls, each a Thought transforming Meanings. The core pipeline is a seven-phase sequence. Optional thinking pattern Thoughts (47 total) inject at three points. None are verbalized — they stay internal. Intermediate result Meanings are data, not instructions — the same architectural move that prevents prompt injection in conversation (Chapter 10) prevents reasoning contamination.

## 10.1 Application: Reasoning

The structured reasoning scaffold decomposes complex thinking into many
small, focused LLM calls. In Meaning/Thought terms: a program applies a
sequence of Thoughts to Meanings, where each Thought transforms input
Meanings into output Meanings. The full trace is a chain of Thoughts.
None of them are verbalized --- they stay internal.

### 10.4.1 Design Properties

**Prompt specialization.** Each Thought has a tightly scoped instruction
purpose-built for a micro-task.

**Observability for free.** Every intermediate result is a Meaning with
a typed entry in the trace.

**Composability.** Thoughts are functions. Swap one out, A/B test two
implementations, skip Thoughts for simple queries.

**Optimized Thought selection.** Which thinking pattern Thoughts to
activate is a learning problem. The reference strategy is a multi-armed
bandit (BOAD, Xu et al. 2025), but the optimizer is pluggable --- any
strategy that learns from traces with credit assignments can drive
selection.

### 10.4.2 Core Pipeline

The seven-phase pipeline is a fixed sequence of Thoughts:

    Request (Meaning)
      ──▶ characterize ──▶ Characterization (Meaning)
      ──▶ extract_info ──▶ Information Inventory (Meaning[])
      ──▶ decompose ──▶ Task List (Meaning[])
      ──▶ ground ──▶ Grounding Audit (Meaning)
      ──▶ execute(task_1) ──▶ Result 1 (Meaning)
      ──▶ execute(task_2) ──▶ Result 2 (Meaning)
      ──▶ execute(task_3) ──▶ Result 3 (Meaning)
      ──▶ format_plan ──▶ Format (Meaning)
      ──▶ synthesize(results) ──▶ Answer (Meaning)

Each arrow is a Thought. Each node is a Meaning. The core Thoughts:

    characterize = Thought(
        "characterize",
        description="Classify the request's answer form, complexity, and domain.",
        inputs={"request": Meaning("a user request")},
        produces=Meaning("classification of complexity, form, and domain"),
        requirements=["Must classify form and complexity"],
    )

    decompose = Thought(
        "decompose",
        description="Break the request into independent, executable tasks.",
        inputs={"request": Meaning("a user request"),
                "characterization": Meaning("a classification"),
                "info": [Meaning("a fact")]},
        produces=[Meaning("an independent, executable task")],
        requirements=["Tasks must be independently executable"],
    )

    execute = Thought(
        "execute",
        description="Execute a single task, producing a concrete finding.",
        inputs={"task": Meaning("a task"),
                "context": [Meaning("a fact")]},
        produces=Meaning("a concrete finding"),
        requirements=["Must not fabricate specific data not in context"],
    )

    synthesize = Thought(
        "synthesize",
        description="Combine all task results into a complete answer.",
        inputs={"results": [Meaning("a finding")],
                "request": Meaning("a user request")},
        produces=Meaning("a complete answer"),
        requirements=["Must reference every input finding"],
    )

### 10.4.3 Thinking Patterns as a Thought Catalog

The reasoning scaffold design doc (v0.5) defines 38 thinking patterns.
An additional 9 patterns grounded in informal logic and argumentation
theory (Toulmin, Walton, Pollock, Dung, pragma-dialectical theory) are
in development --- the first systematic application of argumentation
theory to LLM reasoning systems. Together, the catalog of 47 patterns
becomes a Thought catalog, organized by what they do to Meanings.

The tables below show representative Thoughts per category, not the full
catalog. Full specifications (triggers, failure modes, repair
strategies, costs, injection points) are in the reasoning scaffold
design doc (v0.5, Sections 3.1--3.38).

**Content-Transforming Thoughts** --- take a Meaning, produce a modified
version:

  --------------------------------------------------------------------------------
  Thought                 Meaning Operation                Pattern \#
  ----------------------- -------------------------------- -----------------------
  `critique`              `result.challenge()`             3.4

  `revise`                `result.incorporate(critique)`   3.4

  `compress`              `answer.compress()`              3.38

  `elaborate` /           `result.concretize()`            3.32
  `worked_example`                                         

  `reframe`               `result.reframe(perspective)`    3.5

  `invert`                `request.negate()` → enumerate   3.13
                          failures                         
  --------------------------------------------------------------------------------

**Structure-Changing Thoughts** --- change the shape of reasoning:

  -----------------------------------------------------------------------
  Thought                 What It Does            Pattern \#
  ----------------------- ----------------------- -----------------------
  `decompose`             Meaning → Meaning\[\]   3.1

  `synthesize`            Meaning\[\] → Meaning   3.1

  `eliminate`             Meaning\[\] ×           3.12
                          Constraint\[\] →        
                          Meaning\[\]             

  `abstract`              Move up                 3.14

  `concretize`            Move down               3.14

  `temporal_split`        Split by time horizon   3.15

  `backward_chain`        Goal → prerequisites    3.10
  -----------------------------------------------------------------------

**Evaluation Thoughts** --- assess without transforming:

  ------------------------------------------------------------------------
  Thought                  Output                  Pattern \#
  ------------------------ ----------------------- -----------------------
  `calibrate`              Meaning + confidence    3.11

  `ground`                 Meaning + grounding     3.2
                           flags                   

  `check_contradictions`   Contradiction report    3.31

  `sanity_check`           Plausibility assessment 3.19

  `trace_causal_chain`     Chain with link         3.20
                           assessments             

  `tag_epistemic_status`   Claims with provenance  3.23

  `track_evidence`         Claim-to-source mapping 3.26
  ------------------------------------------------------------------------

**Robustness Thoughts** --- test stability under perturbation:

  -----------------------------------------------------------------------
  Thought                 Perturbation            Pattern \#
  ----------------------- ----------------------- -----------------------
  `counterfactual`        `assumption.negate()` → 3.34
                          re-derive               

  `stress_test`           Vary parameters → check 3.29
                          survival                

  `cross_validate`        Independent derivation  3.33

  `consensus`             Re-run with variation   3.21

  `red_team`              Parallel opposition     3.36
  -----------------------------------------------------------------------

**Meta Thoughts** --- think about the thinking:

  -----------------------------------------------------------------------
  Thought                 What It Does            Pattern \#
  ----------------------- ----------------------- -----------------------
  `select_strategy`       Enumerate approaches,   3.22
                          choose, log             

  `premortem`             Imagine failure,        3.17
                          identify causes         

  `recycle`               Detect overlap, adapt   3.24
                          prior results           

  `adaptive_depth`        Estimate difficulty,    3.25
                          decompose or mark       
                          atomic                  

  `negotiate_scope`       Narrow focus, select    3.35
                          highest-value subset    
  -----------------------------------------------------------------------

### 10.4.4 Thought Composition

Named Thought sequences for common reasoning profiles:

    high_stakes = ThoughtSequence([
        characterize, assumption_surfacing,
        decompose_execute(with_thoughts=[constraint_propagation]),
        red_team_blue_team, failure_premortem,
        evidence_chain_tracking, confidence_calibration,
        iterative_refinement, contradiction_detection, synthesize,
    ])

    quick_advisory = ThoughtSequence([
        characterize, minimum_viable_answer, confidence_calibration,
    ])

### 10.4.5 Meanings as Data, Not Instructions

Intermediate result Meanings are *data* --- downstream Thoughts use them
as material, not as authoritative instructions. This prevents
self-anchoring, where the model treats its own intermediate output as
ground truth. The same architectural move that prevents prompt injection
in conversation prevents reasoning contamination.

### 10.4.6 Developer Experience

**Minimal:**

    from mellea import start_session
    from mellea.reasoning import reason

    m = start_session()
    answer = reason(m, "How should we migrate our Flask monolith to microservices?")
    print(answer)       # verbalized answer text
    print(answer.trace) # full Thought-by-Thought trace

**Configured:**

    from mellea.reasoning import reason, ReasoningConfig

    config = ReasoningConfig(
        always_use=[critique, confidence_calibration],
        never_use=[consensus_building],
        max_calls=25,
        optimizer="ucb",  # pluggable: "ucb", "contextual_bandit", "llm_selector", ...
    )
    answer = reason(m, request, config=config)

**Custom domain Thought:**

    check_precedent = Thought(
        "check_precedent",
        description="Check consistency with established legal precedent",
        inputs={"conclusion": Meaning("a legal finding")},
        produces=Meaning("a precedent assessment"),
        requirements=["Must cite specific precedents"],
        injection_point="post-execution",
    )

    config = ReasoningConfig(domain_thoughts=[check_precedent])

**Trace inspection:**

    for step in answer.trace:
        print(f"Thought: {step.thought.name}")
        print(f"  Inputs: {[m.intent for m in step.inputs.values()]}")
        print(f"  Output: {step.output.intent}")
        print(f"  Verbalized: {step.verbalized}")
        print(f"  Credit: {step.credit}")

**Post-hoc Meaning manipulation:**

    coupling = answer.trace.find("execute", task="coupling_analysis").output
    executive_version = coupling.compress()
    security_angle = coupling.reframe("security")

### 10.4.7 Latency

In Mode 0, the pipeline runs 15--25 LLM calls sequentially. With
optional thinking patterns, this can reach 30+. Each call takes
\~0.5--2s depending on model and output length, yielding total latencies
of 10--60s for a full reasoning run.

Mitigations, roughly in implementation order:

-   **Complexity routing** (already implemented) skips the full pipeline
    for simple queries, reducing to 2--5 calls.
-   **Parallel task execution.** Tasks with no mutual dependencies can
    run concurrently. The abstraction for this is not yet designed (see
    carried-forward open questions).
-   **Streaming partial results.** The trace is available incrementally;
    a UI can show intermediate Thoughts as they complete.
-   **Mode 1 (LLMON single-stream).** Eliminates per-call context
    reconstruction overhead and enables KV cache reuse, reducing
    wall-clock time even for the same number of logical Thoughts.

For conversation, the latency concern is different: each turn must feel
responsive. The intake pipeline and Thought selection are fast
(deterministic + one LLM call). If the conversation invokes reasoning
(cross-catalog), the user should see a progress indicator while the
reasoning trace executes.

---

> [← Ch.10: Conversation](10-conversation.md) · [Table of Contents](../README.md) · [Ch.12: Unification, Traces, and Synthetic Data →](12-unification-traces-sdg.md)
