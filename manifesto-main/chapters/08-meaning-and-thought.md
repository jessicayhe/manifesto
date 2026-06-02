> [← Ch.7: Mellea Standard Library](07-mellea-stdlib.md) · [Table of Contents](../README.md) · [Ch.9: Decision Architecture →](09-decision-architecture.md)

---

# Chapter 8: Meaning and Thought

**Component:** `mellea.stdlib`
**Status:** Draft
**Version:** 1.1
**Authors:** David Cox, Claude
**Date Created:** 2026-03-24
**Date Modified:** 2026-04-06
**Implementation Target:** Mellea Library

**Prerequisites:** Chapter 7 (Mellea Standard Library — IVR, encapsulation, generative slots), Chapter 2 (LLMON — boundary-constrained masking for Modes 1+)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## A Foundation for Generative Programs that Reason and Converse

This chapter introduces the two primitives of the semantic layer — **Meaning** and **Thought** — and describes how Thoughts work: execution, validation, selection, traces, composition, injection points, and failure modes. Meanings are semantic content independent of surface form; Thoughts are typed cognitive operations over Meanings. Together they provide the programming model for systems that reason (Chapter 11) and converse (Chapter 10) using the same mechanism.

## 8.1 Two Primitives

### 8.1.1 Meaning

A **Meaning** is semantic content independent of surface form. It can be
realized into natural language on demand, differently each time. It
supports transformations: you can negate a Meaning, paraphrase it,
abstract it, concretize it, shift its tone. The result is a new Meaning.

A Meaning's full definition:

    @dataclass(frozen=True)
    class Meaning:
        intent: str                            # semantic description (not a prompt)
        type: Meaning | None = None            # what kind of Meaning (itself a Meaning)
        _operations: tuple = ()                # lazy chain of pending transforms
        degraded: bool = False                 # True if produced by IVR exhaustion
        metadata: dict = field(default_factory=dict)  # extensible annotations

`frozen=True` --- Meanings are immutable. Every operation produces a new
Meaning. The `_operations` tuple accumulates lazily; the full chain is
realized in a single LLM call.

    from mellea.generative import Meaning

    # These are NOT strings. They're semantic objects.
    greeting = Meaning("casual greeting, can help with restaurants")
    coupling = Meaning("auth module has high coupling to payments via shared DB")

    # The same operations work on both — because meaning is meaning,
    # whether it lives in a conversation or a reasoning trace.
    greeting.paraphrase()        # same meaning, different words
    coupling.negate()            # "auth is NOT highly coupled to payments"
    coupling.abstract()          # "the system has tight inter-module coupling"
    coupling.concretize()        # "specifically, auth_service.py imports payments.models..."
    coupling.reframe("security") # same facts, viewed as a security concern
    greeting.in_tone("formal")   # same meaning, different register

A Meaning is only realized into text at the boundary where the system
produces output. Internally, the system operates on Meanings. This is
what makes data isolation possible: user input is parsed *into* Meanings
(stripping surface form), and those Meanings flow through the system as
typed semantic objects, placed in data positions where they cannot be
confused with instructions.

**Honesty about the mechanism.** In Mode 0, a Meaning's `.intent` is a
string. Data isolation is enforced by Mellea's API: Meanings appear in
`user_variables` (data plane), never in `description` (control plane).
This is a strong convention --- the same insight as the security
community's control/data plane separation --- but it is not a structural
guarantee at the token level. Structural enforcement arrives with LLMON
constraint masking (Modes 1--3), where the inference runtime prevents
data spans from being attended to as instructions. The design is
layered: convention in Mode 0, structural in Modes 1+.

#### Meaning operations

    # Intent-preserving transforms (same core meaning, different expression)
    m.paraphrase()          # same meaning, different words
    m.repeat()              # same meaning, same level of detail
    m.compress()            # same meaning, fewer details
    m.elaborate()           # same meaning, more details

    # Intent-altering transforms (different meaning)
    m.negate()              # invert the meaning
    m.abstract()            # move up the abstraction ladder
    m.concretize()          # move down — add specifics
    m.reframe(perspective)  # same facts, different framing
    m.incorporate(other)    # merge another Meaning's content in
    m.challenge()           # identify weaknesses or counterarguments
    m.resolve(issue, info)  # resolve an ambiguity or gap with new information

    # Register transforms
    m.in_tone("formal")
    m.for_audience("expert")

    # Composition
    m.and_(other)           # combine two Meanings

Operations are lazy --- they produce new Meaning objects, not text. A
chain like `m.negate().in_tone("formal")` is a single Meaning that gets
realized in one LLM call, not a pipeline of string rewrites.

**Soft algebra.** Meaning operations have intentional, not formal,
semantics. `m.negate()` does not compute logical negation; it produces a
Meaning whose realization is an LLM's best attempt at negation. There is
no guarantee that `m.negate().negate() ≈ m`, or that
`m.abstract().concretize()` recovers `m`. This is by design: the
operations are cognitive heuristics, not functions over a formal domain.
Reliability comes from IVR validation on the Thoughts that consume these
Meanings, not from algebraic properties of the operations themselves.

#### Realization

    # Meaning → text, mediated by an LLM call
    # Each realization is different. The Meaning is the invariant.
    text = greeting.realize(m, context={"time_of_day": "evening"})
    # → "Good evening! Looking for somewhere to eat?"
    # next time:
    # → "Hey! Need help finding a restaurant tonight?"

**Realization vs. verbalization.** `realize()` converts a Meaning to
text. `verbalize()` is `realize()` that crosses the boundary into speech
--- the output is sent to the user. Internally they are the same
operation; verbalization is realization with a side effect (the text
enters the conversation). The `verbalized` flag on a TraceEntry records
whether this happened.

### 8.1.2 Types as Meanings

The type of a Meaning is itself a Meaning. This makes the type system
semantic, not structural --- types support the same operations as any
Meaning, and type matching is a judgment about meaning, not an
`isinstance()` check.

    # Types are Meanings that describe kinds of meaning
    result_type = Meaning("a concrete finding from executing a specific task")
    critique_type = Meaning("an argument against a result, identifying specific weaknesses")
    question_type = Meaning("a request for information from the user")

    # A Meaning has a type, which is a Meaning
    coupling = Meaning(
        "auth module has high coupling to payments via shared DB",
        type=result_type,
    )

    # Types undergo the same operations as any Meaning
    security_result_type = result_type.concretize("specifically about security implications")
    assessment_type = critique_type.abstract()  # "any evaluative judgment"
    grounded_result_type = result_type.and_(Meaning("backed by evidence from provided input"))

Type matching is soft --- a spectrum, not a binary:

    # A "preliminary finding" is close enough to a "result"
    preliminary = Meaning("auth seems coupled", type=Meaning("a preliminary finding"))
    critique(preliminary)  # accepted: preliminary finding ≈ result

    # A "user greeting" is nothing like a "result"
    greeting = Meaning("hello", type=Meaning("a conversational greeting"))
    critique(greeting)  # rejected: greeting ≇ result

This softness means: - **Types are discoverable.** A new kind of
artifact is just a new Meaning. No schema migration, no class
hierarchy. - **Types are transformable.** Specializing, generalizing,
and composing types works the same as for any Meaning. - **Type checking
is a semantic question.** "Is this suitable input for critique?" is a
question the system can *reason about*.

The constraint: softness must not be so soft that it's useless. Thoughts
declare their input types as Meanings, and matching uses a configurable
threshold --- strict for safety-critical boundaries, soft for creative
composition.

**Reference implementation.** The default matcher embeds both the input
Meaning's type and the expected type using the session model's
embeddings, and compares cosine similarity against the threshold. This
is fast (no additional LLM call) and sufficient for the common case
where type names are descriptive. For high-stakes boundaries (safety
gate, cross-catalog invocation), an LLM judge provides higher fidelity
at the cost of one call per check. The matcher is a strategy on the
Thought, defaulting to embedding similarity:

    critique = Thought(
        "critique",
        ...
        type_matcher="embedding",       # default: fast, usually sufficient
        type_accept=0.7,                # above: accept
        type_reject=0.4,                # below: reject; between: warn and proceed
    )

    safety_gate = Thought(
        "safety_check",
        ...
        type_matcher="llm_judge",       # expensive, high fidelity
        type_accept=0.9,                # narrow ambiguous zone for safety
        type_reject=0.8,
    )

### 8.1.3 Thought

A **Thought** is a cognitive operation over Meanings. It takes Meanings
as input, produces Meanings as output, and carries its own requirements
and validation. It's the unit of *thinking* in the system --- whether
that thinking eventually gets verbalized as conversation or stays
internal as reasoning.

A Thought's full definition:

    @dataclass
    class Thought:
        name: str                              # unique identifier
        description: str                       # control-plane instruction
        inputs: dict[str, Meaning]             # named input types (as Meanings)
        produces: Meaning | list[Meaning]      # output type (as Meaning)
        requirements: list[str]                # IVR requirements
        ivr_strategy: IVRStrategy = "default"  # deterministic, check, repair
        type_matcher: str = "embedding"        # "embedding" or "llm_judge"
        type_accept: float = 0.7              # above this: accept match
        type_reject: float = 0.4              # below this: reject (TypeMismatch)
                                               # between: accept with warning
        injection_point: str | None = None     # "pre-execution", "post-execution", "post-synthesis"
    from mellea.generative import Meaning, Thought

    # A Thought: typed cognitive operation + requirements + validation
    elicit = Thought(
        "elicit",
        description="Determine what information is needed from the user",
        inputs={"target": Meaning("an information need")},
        produces=Meaning("a question"),
        requirements=["single sentence", "preference for minimization"],
    )

    critique = Thought(
        "critique",
        description="Find weaknesses in a result",
        inputs={"target": Meaning("a concrete finding")},
        produces=Meaning("an argument identifying specific weaknesses"),
        requirements=["must identify specific weaknesses, not vague concerns"],
    )

    # A Thought applied to Meaning(s) → new Meaning
    # (Simplified notation — the MelleaSession is bound implicitly
    # when Thoughts execute inside a pipeline or conversation.
    # The full signature is in Section 2.1.)
    question = elicit(cuisine_preference)      # Meaning → Meaning
    problems = critique(coupling_analysis)     # Meaning → Meaning
    answer   = synthesize(task_1, task_2)      # Meaning, Meaning → Meaning

The difference between reasoning and conversation is not what a Thought
*is* --- it's whether the Thought gets **verbalized**. A critique stays
internal; an elicitation gets verbalized as a question. A decomposition
stays internal; an assertion gets verbalized as a statement.
Verbalization is what happens when a Thought crosses the boundary into
speech.

    # Internal: Thought produces a Meaning that stays in the trace
    problems = critique(coupling_analysis)

    # Verbalized: Thought produces a Meaning that becomes speech
    question_text = elicit(cuisine_pref).verbalize(m, context)

Thoughts are the shared mechanism between conversation and reasoning.
The catalogs differ --- conversational thoughts include *elicit*,
*repair*, *close*; reasoning thoughts include *critique*, *eliminate*,
*probe* --- but the mechanism is identical: select a Thought based on
state, bind Meanings to its inputs, execute under requirements, capture
the output Meaning.

## 8.2 How Thoughts Work

### 8.2.1 Execution

A Thought executes by:

1.  **Type check.** Verify input Meanings are compatible with declared
    input types (soft match).
2.  **Bind.** Insert input Meanings into the Thought's instruction as
    `user_variables`.
3.  **Generate.** Call `m.instruct()` with the Thought's own instruction
    and requirements.
4.  **Validate.** Check output via Mellea's instruct-validate-repair
    loop.
5.  **Capture.** Tag the output as a Meaning with the declared output
    type.

The critical data isolation property: input Meanings appear in
`user_variables` (the data plane), never in `description` (the control
plane). The Thought's instruction is the control plane.

    class Thought:
        def __call__(self, m: MelleaSession, **kwargs: Meaning) -> Meaning:
            # 1. Soft type check (three-zone: accept / warn / reject)
            for name, meaning in kwargs.items():
                expected = self.inputs[name]
                score = meaning.type.similarity(expected, matcher=self.type_matcher)
                if score < self.type_reject:
                    raise TypeMismatch(meaning.type, expected, score)
                if score < self.type_accept:
                    log.warning(f"Ambiguous type match: {meaning.type} ≈ {expected} ({score:.2f})")
            
            # 2-3. Bind and generate
            result = m.instruct(
                description=self.description,
                user_variables={k: v.intent for k, v in kwargs.items()},
                requirements=self.requirements,
                strategy=self.ivr_strategy,
            )
            
            # 4-5. Validate and capture
            return Meaning(result.value, type=self.produces)

**Context visibility.** In Mode 0, each Thought sees only its declared
inputs (as `user_variables`) and its own description (as the system
prompt). A Thought does *not* see the full accumulated trace, other
Thoughts' outputs, or the raw user request --- unless these are
explicitly passed as inputs. This is the default and the recommended
mode: it minimizes context window usage and prevents Thoughts from
anchoring on upstream artifacts they weren't designed to consume.

The pipeline orchestrator is responsible for binding the right Meanings
to each Thought's inputs. For example, `synthesize` receives all task
results because its input type is `[Meaning("a finding")]`, and the
orchestrator collects the outputs of all `execute` calls into that list.
The original request is passed explicitly to Thoughts that declare it as
an input (e.g., `characterize`, `decompose`, `synthesize`), not
broadcast to all Thoughts.

In Modes 1--3, LLMON constraint masking enforces the same visibility
structurally: a Thought span can only attend to its declared input
result spans, not the full trace.

### 8.2.2 Validation (IVR)

Mellea's instruct-validate-repair loop is the execution engine for
Thoughts. Three layers:

  -----------------------------------------------------------------------
  Layer                   Mechanism               Example
  ----------------------- ----------------------- -----------------------
  Deterministic           Python checks, instant  Is it a numbered list?
                                                  Are all tasks
                                                  referenced?

  `check()` anti-priming  Requirement excluded    "Do not fabricate data"
                          from prompt, validated  
                          after                   

  Targeted repair         Re-prompt with specific "Fix these problems:
                          failures                \[list\]"
  -----------------------------------------------------------------------

Each Thought carries its own IVR strategy.

### 8.2.3 Thought Selection as Decision

Given a set of available Thoughts and the current state, something must
choose which Thought to apply next. Following Sun (2025), we formalize
this as an explicit decision problem with three components:

**Decision point** `(A, c, δ)`**.** At each point where the system must
choose a Thought:

-   **Actions** `A`: the set of Thoughts legal at this point
    (constrained by grammar and injection point).
-   **Decision context** `c`: the Meanings accumulated so far ---
    characterization, info inventory, prior results, IVR outcomes,
    budget remaining, and any explicit decision signals.
-   **Decision function** `δ(c) → a`: a deterministic function that maps
    context to an action. Given the same exposed context, `δ` selects
    the same Thought.

The key property is **separation**: the signals that inform the decision
are computed and logged independently from the policy that acts on them.
This enables **attribution** --- when the system makes a bad Thought
selection, the cause can be localized to:

1.  **Signal error**: the decision context was wrong (e.g., the
    characterization misjudged complexity)
2.  **Policy error**: the signals were correct but the policy chose the
    wrong Thought (e.g., the optimizer selected critique when calibrate
    would have been more useful)
3.  **Execution error**: the right Thought was selected but executed
    poorly (e.g., IVR failed to catch a bad output)

This three-way attribution is impossible when selection is implicit.

**Decision signals.** The decision context `c` contains explicit signals
extracted from the accumulated Meanings:

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

The **sufficiency** signal reflects whether the current context contains
enough information to act reliably --- whether to execute the next
pipeline phase, elicit another slot, or expand a retrieval set. The
**correctness** signal reflects whether the current trajectory appears
to be on track --- whether the last Thought's output passed IVR, whether
task results are consistent, whether confidence is high enough.

These signals can be: - **Rule-based**: slot frame completeness, budget
exhaustion - **LLM-estimated**: an LLM judge assessing sufficiency -
**Embedding-based**: similarity between accumulated context and expected
output type - **Composite**: weighted combination of multiple signal
sources

The signals are **modular** --- you can swap the signal estimator
without changing the policy, or swap the policy without changing the
signals. This is the same property that makes the optimizer pluggable,
but now formalized.

**Three selection mechanisms compose as decision layers:**

**Grammar-based constraint** `A → A'`**.** Narrows the action set before
the decision function runs. The NCF grammar constrains conversation
Thoughts; the pipeline phase ordering constrains core Thoughts; the
injection point constrains pattern Thoughts.

**Pragmatic override.** If a slot is unfilled, the action is `elicit`
regardless of what the optimizer would choose. If IVR just failed, the
action is `repair` or `clarify` (no-blind-retry constraint). These are
hard rules applied before the optimizer.

**Optimizer** `δ(c) → a`**.** Selects among the remaining candidates.
Implementations:

  -----------------------------------------------------------------------
  Strategy                How it works            When to use
  ----------------------- ----------------------- -----------------------
  Static                  Fixed list per profile  Bootstrapping, SDG
                          (e.g., high_stakes)     

  Threshold               Act if                  Simple act-vs-clarify
                          `sufficiency > τ`, else 
                          clarify/skip            

  UCB bandit              Explore/exploit over    Offline optimization
                          Thought arms            

  Contextual bandit       Condition on            Personalized selection
                          characterization        
                          features                

  LLM-as-selector         Ask an LLM which        Flexible, expensive
                          Thought to apply        

  Learned policy          RL over trace rewards   End-to-end optimization
  -----------------------------------------------------------------------

All share the same interface: `δ(c) → Thought`. The grammar constrains
what's *legal*, pragmatic rules determine what's *required*, and `δ`
selects what's *helpful* among the remaining options.

**Decision logging.** Every decision is recorded in the trace alongside
the Thought execution:

    @dataclass
    class DecisionRecord:
        available: list[str]         # Thought names in A
        context_signals: dict        # {"sufficiency": 0.75, "correctness": 0.9, ...}
        policy: str                  # which δ was used
        selected: str                # which Thought was chosen
        rationale: str | None        # optional: why (for LLM-as-selector)

This makes every Thought selection inspectable after the fact. Combined
with the TraceEntry's validation and credit fields, the full chain is:
context → signals → decision → execution → validation → credit.

**Asymmetry between catalogs.** In reasoning, the optimizer is central
--- it selects among 30+ optional thinking patterns. In conversation,
selection is currently deterministic: the NCF grammar and pragmatic
state fully determine the next Thought. Whether conversation Thought
selection should eventually be learned (e.g., which elicitation strategy
to use, when to batch vs. eagerly elicit, which repair to attempt) is an
open question. The decision-centric formulation supports this directly
--- the same `(A, c, δ)` interface applies to both catalogs.

### 8.2.4 Traces

Every Thought execution is recorded, along with the decision that
selected it:

    @dataclass
    class TraceEntry:
        thought: Thought              # which Thought was applied
        inputs: dict[str, Meaning]    # input Meanings (by name)
        output: Meaning               # output Meaning
        validation: ValidationResult  # which requirements passed/failed
        verbalized: bool              # did this cross into speech?
        credit: float | None          # optimizer credit (post-hoc)
        decision: DecisionRecord | None  # how this Thought was selected

A trace is a list of TraceEntries. It's the full record of how a
reasoning chain or conversation unfolded. Because every entry is typed
(Thought + Meanings), the trace is structured, inspectable, and
replayable.

### 8.2.5 Thought Composition

Thoughts can be composed into higher-level Thoughts:

    # Adversarial Self-Critique is critique + revise
    adversarial_self_critique = critique.then(revise)

    # Conditional composition
    critique_if_needed = critique.gate(
        condition=lambda meaning: meaning.type.matches(Meaning("a complex finding")),
        fallback=identity,
    )

`then` and `gate` compose two Thoughts into a new Thought --- the result
is itself a Thought with typed I/O that can be used anywhere a single
Thought can. Composition checks type compatibility: `a.then(b)` verifies
that `a.produces` is compatible with the first input of `b` using the
soft type matcher at composition time. If the match falls in the
ambiguous zone, composition succeeds with a warning; below the reject
threshold, it raises `CompositionError`. The composed Thought inherits
its inputs from `a` and its `produces` from `b`.

`ThoughtSequence` (Section 4.4) is a higher-level construct: a named
pipeline profile that lists which Thoughts to activate across injection
points. A `ThoughtSequence` is not a Thought --- it's a configuration
that the pipeline executor and optimizer use to select and order
Thoughts within a reasoning run.

### 8.2.6 Injection Points

The core reasoning pipeline (Section 4.2) defines three points where
optional Thoughts can be inserted:

  -----------------------------------------------------------------------
  Injection Point         When                    Examples
  ----------------------- ----------------------- -----------------------
  **pre-execution**       After decompose, before assumption surfacing,
                          task execution          socratic probing,
                                                  elimination, backward
                                                  chaining

  **post-execution**      After each task         critique,
                          executes                multi-perspective,
                                                  constraint propagation,
                                                  confidence calibration

  **post-synthesis**      After final synthesis   iterative refinement,
                                                  contradiction
                                                  detection, compression
  -----------------------------------------------------------------------

A Thought declares its injection point as part of its definition. The
optimizer selects which optional Thoughts to activate at each point; the
pipeline invokes them in the declared order. Conversation has analogous
insertion points defined by the NCF grammar (pre-response,
post-response, pre-close).

### 8.2.7 Failure and Degradation

Every Thought execution can fail. The system degrades gracefully:

**IVR exhaustion.** If a Thought's IVR loop exhausts all repair
attempts, the Thought returns a *degraded Meaning* --- the best output
achieved, with `degraded=True`. Downstream Thoughts can check this flag
and decide whether to proceed or skip. The trace records the validation
failure separately via `TraceEntry.validation`.

**Type mismatch.** If soft type matching returns an ambiguous score
(between reject and accept thresholds), the system logs a warning and
proceeds with the match. Only scores below the reject threshold raise
`TypeMismatch`. This avoids brittle failures on borderline types while
preserving hard boundaries.

**Optimizer uncertainty.** If the optimizer cannot distinguish among
candidates (e.g., cold start, insufficient data), it falls back to a
static default set defined per pipeline profile. The trace records that
fallback was used, providing signal for future optimization.

**Meaning operation failure.** If realization produces nonsense
(detected by an optional output validator on the Meaning), the system
retries with a fresh LLM call. After N retries, it returns the input
Meaning unchanged --- a no-op is safer than a bad transform.

**Cross-catalog suspension.** When reasoning invokes a conversation
Thought (e.g., `yield elicit()`), the reasoning state is serialized to a
`SuspendedTrace` and the session yields to the conversation loop. If the
user never responds (timeout), the reasoning trace resumes with a
`Meaning("no response received")` and the Thought that requested
clarification must handle this case in its requirements.

---

> [← Ch.7: Mellea Standard Library](07-mellea-stdlib.md) · [Table of Contents](../README.md) · [Ch.9: Decision Architecture →](09-decision-architecture.md)
