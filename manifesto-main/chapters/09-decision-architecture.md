> [← Ch.8: Meaning and Thought](08-meaning-and-thought.md) · [Table of Contents](../README.md) · [Ch.10: Conversation →](10-conversation.md)

---

# Chapter 9: Decision Architecture

**Component:** `mellea.stdlib`  
**Status:** Draft v1.0  
**Authors:** Wei Sun, David Cox
**Date:** 2026-04-06

**Prerequisites:** Chapter 8 (Meaning and Thought primitives, decision preview in §8.2.3)


---

## How Systems Choose What to Think Next

Chapter 8 introduced decision points as the mechanism for Thought selection: given available Thoughts and accumulated Meanings, something must choose what to think next. This chapter develops the full decision architecture. It formalizes the decision problem, defines the signal and policy interfaces, specifies the three selection layers that compose to govern Thought selection, and provides the logging and attribution machinery that makes every decision inspectable. 

## 9.1 The Decision Problem

A decision arises wherever the system must choose which Thought to apply next. Rather than leaving this choice implicit inside a single LLM generation call, we expose it as an explicit, inspectable layer with three components.

### 9.1.1 Decision Points

A **decision point** is a tuple `(A, c, δ)`:

-   **Actions** `A`: the set of candidate Thoughts available at this point.
-   **Decision context** `c`: the structured state available when the
    decision is made.
-   **Policy** `δ`: the function that maps context to a chosen action.

This formalization makes action selection explicit and repeatable. Given the same exposed context, the same policy always selects the same Thought.

The purpose of the separation is *attribution*. When selection is implicit, i.e., fused inside a single model call, a bad choice is an opaque model behavior. When selection is explicit, failures can be localized to context construction, signal estimation, policy logic, or execution quality (§9.4).

### 9.1.2 Actions as Thoughts

In Mellea, actions are Thoughts. The action set `A` at any decision point is the set of Thoughts legal at that point, constrained by the grammar (§9.3.1) and the current injection point. Each Thought carries its type signature, requirements, and IVR strategy (Chapter 8, §8.2.1--8.2.2), so the action set is typed and self-describing.


### 9.1.3 Decision Context

The **decision context** `c` is the full structured state available at a decision point. It includes the Meanings accumulated so far, prior outcomes, budget constraints, and any explicit signals derived from these.

    @dataclass
    class DecisionContext:
        available_thoughts: list[Thought]     # A: legal at this point
        characterization: Meaning             # from Phase 1
        accumulated_results: list[Meaning]    # from prior Thoughts
        budget_remaining: int                 # calls left
        sufficiency: float                    # 0-1: enough info to proceed?
        correctness: float                    # 0-1: is the trajectory on track?
        ivr_outcomes: list[bool]              # recent validation results
        metadata: dict                        # extensible

This is the same dataclass introduced in §8.2.3. The `sufficiency` and `correctness` fields are **optional** --- they are common signal patterns (§9.2.2) that recur across domains, not structural requirements. An application that does not need them can leave them at a default value or ignore them entirely. A policy can read these fields or operate directly on the accumulated Meanings.

The dataclass above captures the core fields shared across applications. Additional state --- conversation history, retrieved passages, candidate pools, domain-specific features --- lives in the `metadata` dict or in application-specific subclasses. The context is constructed fresh at each decision point by the application, not mutated in place. The decision function reads this context; it does not maintain any internal state across calls.

### 9.1.4 The Policy

The **policy** `δ(c) → a` is the function that maps a decision context to a chosen action. It is the decision rule --- the mechanism that converts observation into action.

The policy is a **stateless function**. Given the same `DecisionContext`, it always returns the same Thought. The workflow around it may be stateful, i.e., the application maintains conversation history, candidate pools, accumulated results, and evolves the context across turns, but the decision rule itself has no internal memory.

This stateless/stateful separation is a key design property:

-   **Reproducible**: replay the same context, get the same action.
-   **Testable**: unit-test the policy in isolation from the workflow.
-   **Inspectable**: the logged context fully explains the action.

**Note on learned policies.** Strategies like UCB bandits and contextual bandits maintain arm statistics across calls, they are inherently stateful learners. The resolution is that their learned state (arm counts, reward estimates, model weights) lives in the trace store or an external model, not inside the policy function. The policy *reads* those statistics from the context or model at decision time, but given the same inputs it still returns the same action. The learning happens offline, between calls, by updating the external model from trace data. The policy function itself remains stateless; the statefulness of the learner is externalized.

The policy can take many forms:

-   **Rule-based**: if sufficiency is below threshold, clarify; otherwise
    execute.
-   **Optimization-based**: score each candidate Thought by an objective
    and pick the best. One common example is a reward--cost tradeoff:
    `U(a, c) = R(a, c) − Σ_k λ_k C_k(a, c)`, where `R` is a reward
    signal and `C_k` are cost terms (latency, API cost, token budget).
    This is useful for model routing or compute allocation, but it is
    only one possible objective formulation. Others include regret-based,
    constraint-based, ranking-based, or SLA-driven objectives.
-   **Learned**: a bandit or RL policy trained on trace data with credit
    assignments.
-   **LLM-based**: ask an LLM which Thought to apply --- flexible but
   could be expensive and less predictable.

All share the same interface: `δ(c) → Thought`. The strategy is pluggable (§9.5); the interface is fixed.



## 9.2 Decision Signals

Signals sit between context and policy as an **optional** derived layer. A signal is a named, decision-relevant summary extracted from the raw context, that is, analogous to engineered features or summary statistics that facilitate decision-making. A policy can always read the full `DecisionContext` directly. Signals are a convenience that makes the basis for action selection explicit and loggable, not a required architectural component.

### 9.2.1 The Signal Interface

A signal transforms raw context (conversation history, accumulated Meanings, candidate pools, retrieved passages) into a compact quantity that a policy can branch on or a scoring function can use.

Three forms:

1.  **Static value** --- precomputed and passed in (e.g., `0.75`).
2.  **Context-level callable** --- derives a summary from the full
    context: `(DecisionContext) → float`. Evaluated once before the
    policy runs.
3.  **Per-action callable** --- computes a value for each candidate
    Thought: `(Thought, DecisionContext) → float`. Used when the policy
    needs to score each candidate (e.g., in optimization-based
    selection).

Signals provide three benefits when used: (1) they make the basis for action selection explicit and loggable, (2) they decouple the "what do I observe?" step from the "what do I do?" step, enabling independent improvement of each, and (3) they are the natural place to plug in learned estimators (LLM judges, embedding models, calibrated classifiers).

### 9.2.2 Common Signal Patterns

Two signal patterns recur often enough across domains to be worth describing as examples. Neither is required by the architecture.

**Sufficiency** (`sufficiency: float`, 0--1) reflects whether the current context contains enough information to act reliably. In conversation, this might be slot-frame completeness (3 of 4 required fields confirmed → 0.75). In retrieval, it might be the confidence that retrieved passages contain the answer. In reasoning, it might be whether enough sub-tasks have been executed to synthesize.

Signal sources for sufficiency:

-   **Rule-based**: count confirmed fields / total required fields.
-   **LLM-estimated**: ask an LLM judge whether the passages are
    sufficient to answer.
-   **Embedding-based**: cosine similarity between the question and
    passage embeddings.
-   **Composite**: weighted combination of multiple sources.

**Correctness** (`correctness: float`, 0--1) reflects whether the current trajectory appears to be on track. Low correctness after an attempted action favors revision or backtracking. In reasoning, this might be IVR pass rate across recent Thoughts. In graph search, it might be structural consistency of a visited node against the candidate pool.

**Budget** (`budget_remaining: int`) is a hard constraint, not a soft signal. When budget reaches zero, the pragmatic override (§9.3.2) fires regardless of other signal values.

**IVR outcomes** (`ivr_outcomes: list[bool]`) are the recent validation results from Thought executions. A string of failures may indicate the trajectory has gone wrong, even if the correctness signal has not yet caught up.

### 9.2.3 Signal Modularity

A key property of the architecture: **signals and policies are independently improvable**. You can swap the sufficiency estimator without changing the policy, or swap the policy without changing the signals. Performance differences then isolate signal quality from control logic.

This modularity means that when a decision is wrong, the diagnosis starts with a clean separation: was the signal wrong (the system misjudged the situation), or was the policy wrong given a correct signal (the system knew the situation but chose the wrong action)?

This is the same property that makes the optimizer pluggable (§9.5), but applied at the signal level. New signal sources can be introduced, compared, and evaluated under the same policy without changing the decision logic.

### 9.2.4 Practical Challenges

While signals improve inspectability, they introduce their own difficulties:

-   **Signals are estimates, not ground truth.** Many signals (sufficiency,
    correctness, answerability) are noisy or learned approximations. A
    wrong signal can cause a wrong action, and distinguishing signal error
    from policy error is not always straightforward --- both produce the
    same observable outcome (a wrong action) but require different fixes.
-   **Context construction dominates difficulty.** The decision rule
    assumes a well-formed context. In practice, extracting structured
    state from language, maintaining it across turns, and handling partial
    or invalid updates is the hardest part. A correct policy cannot
    compensate for a poorly constructed context.
-   **Signal design is a modeling problem.** Choosing which aspects of the
    context to expose as signals requires domain knowledge. A signal that
    is too coarse may not capture enough information; one that is too
    fine-grained may be noisy or expensive to compute.

## 9.3 The Three Selection Layers

Thought selection is not a single decision --- it is a pipeline of three layers that compose to narrow and select. Each layer has a distinct role, and together they separate what is **legal**, what is **required**, and what is **helpful**.

### 9.3.1 Grammar Constraint (A → A')

The grammar layer is a structural filter that narrows the action set before the policy runs. It answers the question: **what is legal here?** No optimizer can override it.

A grammar defines which Thoughts are structurally available at each point in a workflow. In conversation, the NCF grammar (Chapter 10) plays this role. In reasoning, the pipeline phase ordering and injection points (Chapter 11) play the same role. The specific grammars are defined in their respective chapters; the decision architecture requires only that such a filter exists as the first stage of the composition `A → A'`.

### 9.3.2 Pragmatic Override

After grammar filtering, **pragmatic overrides** apply hard rules before the optimizer runs. These encode domain invariants that no optimizer should be allowed to violate.

Examples:

-   **Unfilled slot → elicit.** If a required slot is empty, the action
    is `elicit` regardless of what the optimizer would choose. The system
    must collect the missing information before proceeding.
-   **Failed execution → clarify (no-blind-retry).** If the last action
    was `execute` and validation failed, the action is `clarify` or
    `repair` --- never a second blind execution. This prevents wasting
    turns on repeated failures without acquiring new information. This is referred to as a "structural constraint
    enforced at the decision layer" in Sun (2026).
-   **Budget exhausted → terminate or degrade.** If `budget_remaining`
    reaches zero, the system must stop or produce a best-effort answer.
    No optimizer can override a hard budget constraint.
-   **IVR exhaustion → escalate.** If IVR has failed repeatedly on the
    same Thought, repair is unlikely to help. The pragmatic layer can
    force a skip or degraded output.

Pragmatic overrides are **structural guarantees**, not prompt-level requests. A prompt-based system cannot ensure that a model will never retry blindly after failure --- it can only ask. The pragmatic layer enforces this as a hard rule that fires before the optimizer is consulted.

### 9.3.3 Optimizer (δ)

If no pragmatic override fires, the **optimizer** `δ(c) → a` selects among the remaining candidates. The optimizer is pluggable --- different strategies can be swapped without changing the grammar or pragmatic layers.

Strategies range from static Thought lists (fixed sequences, no learning) through rule-based policies (threshold on sufficiency, multi-signal branching) to optimization-based and learned policies (reward--cost scoring, bandits, RL from trace credit). All share the interface: `δ(c) → Thought`. The full strategy menu and progressive complexity path are described in §9.5.

As one illustrative example: when the decision involves resource tradeoffs (e.g., model routing under latency constraints), the optimizer can score each candidate by:

    U(a, c) = R(a, c) − Σ_k λ_k C_k(a, c)

where `R` is a reward signal, `C_k` are cost terms, and the optimizer picks the highest-scoring Thought from the feasible set. This is only one possible objective --- regret-based, constraint-based, and ranking-based objectives are equally valid.

The grammar constrains what is *legal*, pragmatic rules determine what is *required*, and `δ` selects what is *helpful* among the remaining options.

### 9.3.4 Layer Composition

The three layers compose as a pipeline:

    A ──grammar──▶ A' ──pragmatic──▶ a | A'' ──optimizer──▶ a

-   If the grammar leaves only one candidate, neither pragmatic nor
    optimizer matters.
-   If a pragmatic override fires, it returns an action directly --- the
    optimizer is not consulted.
-   If no override fires, the optimizer selects from the remaining set.

This composition is the same for conversation and reasoning. Only the grammar rules and the pragmatic conditions differ. The optimizer interface is shared.

## 9.4 Decision Logging and Attribution

Every decision is recorded. The purpose is not just audit --- it is to enable the three-way attribution that makes the architecture diagnosable.

### 9.4.1 The Decision Record

Each decision produces a `DecisionRecord` that captures the full selection process, making the three-layer composition inspectable. When a pragmatic override fires, the record shows which override was triggered and that the optimizer was not consulted.

    @dataclass
    class DecisionRecord:
        available: list[str]             # Thought names in A
        grammar_filtered: list[str]      # Thought names in A' after grammar
        pragmatic_override: str | None   # if pragmatic fired, which Thought
        context_signals: dict            # {"sufficiency": 0.75, "correctness": 0.9, ...}
        policy: str                      # which δ was used
        selected: str                    # which Thought was chosen
        rationale: str | None            # optional: why (for LLM-as-selector)


### 9.4.2 Three-Way Attribution

When the system makes a bad Thought selection, the cause can be localized:

1.  **Signal error**: the decision context was wrong. The sufficiency
    signal read 0.95 but should have been 0.3 --- the system thought it
    had enough information when it did not. Fix: improve the signal
    estimator.
2.  **Policy error**: the signals were correct but the policy chose the
    wrong Thought. The sufficiency was genuinely low, but the optimizer
    selected execute instead of clarify. Fix: adjust the policy or
    threshold.
3.  **Execution error**: the right Thought was selected but executed
    poorly. The system correctly chose synthesize, but the LLM produced a
    hallucinated answer that IVR failed to catch. Fix: improve the
    Thought prompt or IVR strategy.

This three-way attribution is impossible when selection is implicit --- when assessment and action are fused inside a single generation call, a wrong outcome offers no handle for diagnosis.

The `DecisionRecord` provides the raw material for attribution. The `context_signals` field shows what the system believed; the `selected` field shows what it did; the downstream `TraceEntry.validation` field (§8.2.4) shows what happened. Together, they localize the failure.

### 9.4.3 The Full Audit Chain

Every Thought execution participates in a complete audit chain:

    context → signals → decision → execution → validation → credit

-   **Context**: the `DecisionContext` snapshot at this decision point.
-   **Signals**: the derived values (sufficiency, correctness, etc.) at
    decision time.
-   **Decision**: the `DecisionRecord` capturing which Thought was
    selected and why.
-   **Execution**: the Thought's output Meaning.
-   **Validation**: the `ValidationResult` from IVR (§8.2.2).
-   **Credit**: the post-hoc credit assignment for optimizer learning
    (§8.2.4).

This chain is recorded in the trace. The `TraceEntry` dataclass (§8.2.4) includes `decision: DecisionRecord | None`, linking every executed Thought to the decision that selected it. For optimization, the `credit` field feeds back to the optimizer, enabling learning from outcomes.

## 9.5 The Pluggable Optimizer Interface

The decision architecture is designed for progressive complexity: start with a fixed Thought sequence and add decision intelligence incrementally.

### 9.5.1 Interface Definition

Every optimizer implements the same interface:

    Optimizer = Callable[[DecisionContext, list[Thought]], Thought]

Given the filtered action set and the current context, return a Thought. The grammar and pragmatic layers have already run --- the optimizer sees only the candidates that are both legal and not overridden.

Configuration at the pipeline level:

    pipeline = ReasoningPipeline(
        optimizer=ThresholdOptimizer(sufficiency_threshold=0.8),
    )

Or, for more sophisticated selection:

    pipeline = ReasoningPipeline(
        optimizer=BanditOptimizer(model=ucb_model, traces=trace_store),
    )

Or, for cost-aware routing:

    pipeline = ReasoningPipeline(
        optimizer=RewardCostOptimizer(
            reward=quality_model,
            costs={"latency_ms": latency_model, "dollars": cost_table},
            hard_constraints={"latency_ms": ("<=", 800)},
        ),
    )

The pipeline does not know which strategy is active. It calls `optimizer(context, candidates)` and gets a Thought back.

### 9.5.2 Progressive Complexity

Developers should start simple and add decision intelligence incrementally. Each level uses the same `Optimizer` interface --- moving up requires no changes to the grammar, pragmatic rules, Thought definitions, or trace infrastructure.

-   **Fixed sequence.** No decisions. A static `ThoughtSequence` runs the same Thoughts every time.
-   **Threshold policy.** One signal, one branch. Act if `sufficiency > τ`, else clarify.
-   **Multi-signal policy.** Branch on the joint state of sufficiency and correctness. Three or more actions (clarify, execute, backtrack).
-   **Declarative objective.** No hand-written policy. Score each Thought by a reward--cost objective and pick the best feasible one.
-   **Learned policy.** Train from trace data. UCB bandit, contextual bandit, or RL over Thought arms with credit from downstream validation.

## 9.6 Design Properties

The decision architecture supports four properties that are difficult to achieve when control is implicit.

**Attributable failures.** Because signals, policy, and execution are separated, a wrong decision can be localized to the component that failed. If the system acts too early, the logged `DecisionRecord` shows whether the sufficiency signal was wrong (signal error), the threshold was too low (policy error), or the generation was poor despite a correct decision (execution error). This attribution is the basis for targeted repair: fix the broken component without touching the rest.

**Enforceable constraints.** Pragmatic overrides are structural guarantees. The no-blind-retry rule fires before the optimizer and cannot be bypassed by a high sufficiency score. A prompt-based system has no equivalent mechanism --- it can only instruct the model and hope for compliance. The decision layer makes compliance a property of the architecture, not a property of the model.

**Modular improvement.** Swapping one component (a signal estimator, a policy strategy, a Thought prompt) produces measurable, isolated effects. Different sufficiency estimators can be compared under the same policy. Different policies can be compared under the same signals. This isolates signal quality from control logic and enables systematic iteration.

**Domain-agnostic interface.** The same `(A, c, δ)` interface governs conversation control (which NCF Thought to apply), reasoning selection (which thinking pattern to activate), retrieval stopping (whether to expand the passage set), and model routing (which model to invoke). The actions, signals, and policies differ; the architectural pattern does not.

For experimental validation of these properties across calendar scheduling, graph disambiguation, and retrieval control settings, see Sun (2026).

## 9.7 Asymmetry Between Catalogs

The decision architecture applies uniformly to both conversation and reasoning, but the two catalogs use it differently.

In **reasoning** (Chapter 11), the optimizer is central. The thinking pattern catalog contains 47 optional Thoughts, and which subset to activate on a given query is a genuine selection problem. The optimizer must weigh the expected benefit of each pattern against its cost (additional LLM calls, added latency, risk of degrading the trace). This is the setting where learned policies --- bandits, contextual models, RL from trace credit --- are most valuable.

In **conversation** (Chapter 10), selection is currently **deterministic**. The NCF grammar and pragmatic state fully determine the next Thought: if a slot is unfilled, elicit; if the user requests a repair, repair; if the base pair is complete, close. There is no optimizer to speak of --- the grammar and pragmatic layers handle everything.

Whether conversation Thought selection should eventually be learned is an open question. Candidates include: which elicitation strategy to use for a given slot, when to batch multiple elicitations versus asking eagerly, which repair strategy to attempt first, and how aggressively to confirm versus assume. The decision architecture supports this directly --- the same `(A, c, δ)` interface applies, and an optimizer can be introduced without changing the grammar or pragmatic layers.

## 9.8 Open Problems

The decision architecture makes control explicit but does not eliminate its difficulty. Several challenges remain open.

**Credit assignment in sequential decisions.** In multi-step workflows, early decisions affect later context and outcomes. A premature execution at turn 1 that burns a clarification budget affects everything that follows. Attributing a failure to a specific decision point --- rather than to cumulative context drift --- requires careful logging and potentially counterfactual analysis.

**Non-stationary intent.** In multi-turn interactions, the underlying objective may change rather than simply becoming more specified. Users may revise constraints, correct earlier inputs, or switch tasks entirely. Previously computed signals (confirmed fields, candidate sets) may no longer reflect the current objective. Handling this requires mechanisms --- intent tracking, reset logic, confidence decay --- that sit outside the core decision abstraction but are critical in practice.

**Policy brittleness under distribution shift.** Simple threshold-based or rule-based policies work well in controlled settings but can degrade in noisier environments. Small changes in signal values near a decision boundary can flip the chosen action. Learned policies can overfit to their training distribution.

**Signal correlation and indirect state updates.** Actions may affect multiple aspects of the context simultaneously. A failed graph traversal may lower perceived correctness while simultaneously improving sufficiency by eliminating candidates --- a correlated update that the decision rule sees only indirectly through the updated context. The application must track and propagate these indirect effects correctly; the decision layer sees only the resulting snapshot.

**Context construction dominates difficulty.** The decision rule assumes a well-formed `DecisionContext`. In practice, extracting structured state from unstructured language, maintaining it across turns, and handling partial or contradictory updates is often harder than writing the policy itself.

---

## References

Sun, W. (2026). Decision-Centric Design for LLM Systems. [https://arxiv.org/abs/2604.00414](https://arxiv.org/abs/2604.00414)

---

> [← Ch.8: Meaning and Thought](08-meaning-and-thought.md) · [Table of Contents](../README.md) · [Ch.10: Conversation →](10-conversation.md)
