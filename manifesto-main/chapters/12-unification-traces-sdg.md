> [← Ch.11: Reasoning](11-reasoning.md) · [Table of Contents](../README.md) · [Ch.13: Semantic Operators →](13-semantic-operators.md)

---

# Chapter 12: Unification, Traces, and Synthetic Data

**Component:** `mellea.stdlib`
**Status:** Draft
**Version:** 1.1
**Authors:** David Cox, Claude
**Date Created:** 2026-03-24
**Date Modified:** 2026-04-06

**Prerequisites:** Chapter 8 (Meaning and Thought), Chapter 10 (Conversation), Chapter 11 (Reasoning), Chapter 2 (LLMON)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## The Bridge, LLMON Traces, Synthetic Data, and Forward Trajectory

This chapter establishes that conversation (Chapter 10) and reasoning (Chapter 11) are not different mechanisms — they differ only in whether a Thought gets verbalized. It describes cross-catalog invocation, the four invocation modes, LLMON trace encoding for training, synthetic data generation via systematic Meaning variation, and the open questions and implementation priorities that define the path forward.

## 12.1 The Bridge

### 12.1.1 Why They're the Same Mechanism

  -----------------------------------------------------------------------
  Concept                 Reasoning               Conversation
  ----------------------- ----------------------- -----------------------
  Content                 Meaning (a cognitive    Meaning (a
                          artifact)               conversational content)

  Operation               Thought (critique,      Thought (elicit,
                          eliminate, probe)       assert, repair, close)

  Selection               Pluggable optimizer +   NCF grammar + pragmatic
                          pipeline ordering       state

  Validation              IVR (deterministic +    IVR (same, different
                          check + repair)         requirements)

  Trace                   Thought-by-Thought,     Turn-by-turn, some
                          internal                verbalized

  Data isolation          Results are data        User input parsed to
                          Meanings                data Meanings
  -----------------------------------------------------------------------

The difference between reasoning and conversation is not what a Thought
is. It's whether the Thought gets verbalized. A reasoning trace is a
sequence of Thoughts that stay internal. A conversation is a sequence of
Thoughts where some cross the boundary into speech.

### 12.1.2 Shared Meaning Operations

  -----------------------------------------------------------------------
  Operation               In Reasoning            In Conversation
  ----------------------- ----------------------- -----------------------
  `negate()`              Counterfactual probing  NCF disconfirmation
                          (3.34), Inversion       
                          (3.13)                  

  `paraphrase()`          Clarity in synthesis    "What do you mean?"
                                                  repair

  `compress()`            MVA (3.30), Progressive Preference for
                          Disclosure (3.18)       minimization

  `elaborate()`           Worked Examples (3.32)  Extended Telling (A3)

  `concretize()`          Worked Examples (3.32), "Can you give an
                          Abstraction Laddering   example?"
                          (3.14)                  

  `abstract()`            Principle Extraction    Summarization
                          (3.37), Abstraction     
                          Laddering (3.14)        

  `reframe()`             Multi-Perspective       Adjusting to user model
                          (3.5), Stakeholder Sim  
                          (3.16)                  

  `decompose()`           Task Decomposition      Sub-goal creation
                          (3.1)                   

  `critique()`            Adversarial             (natural extension)
                          Self-Critique (3.4),    
                          Red Team (3.36)         

  `eliminate()`           Elimination Reasoning   Narrowing hypotheses
                          (3.12)                  
  -----------------------------------------------------------------------

These aren't analogies. They're the same Meaning transform
specification, invoked in different contexts. "Same operation" means
same intent and same code path --- not identical behavior, since each
realization is a fresh LLM call (see Section 1.1, Soft algebra).

### 12.1.3 Cross-Catalog Invocation

**Conversation invoking reasoning:**

    @find_restaurant.on_complete
    def recommend(m, result, context):
        # Use reasoning Thoughts for a higher-quality answer
        recommendation = reason(m,
            Meaning("recommend a restaurant matching these preferences"),
            context=result.slots,
            config=ReasoningConfig(
                always_use=[critique, confidence_calibration],
                max_calls=10,
            ),
        )
        # Returns a Meaning — supports paraphrase(), concretize(), etc.
        # NCF repairs work on a *reasoned* answer.
        return recommendation

**Reasoning invoking conversation:**

    # Socratic Probing — resolve ambiguity by asking the user
    @app.on("ambiguous_request")
    def clarify(m, request_meaning, ambiguities, context):
        for ambiguity in ambiguities:
            clarification = yield elicit(ambiguity.meaning)
            request_meaning = request_meaning.resolve(ambiguity, clarification)
        return reason(m, request_meaning)

### 12.1.4 Invocation Modes

The four modes describe who selects the next Thought. They apply to both
conversation and reasoning.


|Mode               | Who selects Thoughts         |  Token stream           |  Requirements    |
|  :-----:          |   :--------:                 |   :------:              |    :-------:     |
|  0: Prompted      |       Python program         |   N separate API calls  |    Any model     |
|  1: LLMON         |External Program (via prefill)|  1 continuous stream    | LLMON-aware model|
|  2: LLMON         | Internal The model           |  1 continuous stream    | Trained on traces|
|  3: Mixed Control |  Model (default), system (override) stream | 1 continuous stream |  Trained on traces |
      

Mode 0 is the starting point. The trace format is the same regardless of
mode --- a sequence of (Thought, input Meanings, output Meaning)
entries. Modes 1--3 encode this trace as LLMON tokens in a single
stream.

## 12.2 LLMON Traces

LLMON (Hind et al., 2026) serializes Meaning/Thought sequences into a
single token stream with enforceable structural boundaries.

### 12.2.1 Mapping

| Meaning/Thought Concept |  LLMON Encoding |
| :-----------------:     |  :-----------:  |
|  Thought selection      |  `reason:exec` binding (control)|
|  Thought inputs         |  `reason:exec.input` references to named Meanings |
|  Meaning content        |  Content within phase span (data) |
|  Output Meaning         |  `reason:result:name` span (data) |
|  Thought boundary       |  `reason:pause` token (yield point)|
|  Meaning type           |   Derivable from enclosing span tag|


### 12.2.2 Data/Instruction Separation

LLMON's core property maps directly onto Meaning/Thought:

-   **Instruction:** Thought selection (exec bindings)
-   **Data:** Meaning content (results, intermediate outputs)
-   **Generation:** Model output within Thought spans

Constraint masking enforces that data Meanings are never treated as
instructions.

### 12.2.3 Conversation Traces

The same LLMON encoding works for conversation. Conversational Thoughts
would get their own tag vocabulary under a `conv` prefix --- an
application-level extension of LLMON's tag system, to be co-designed
with the LLMON authors (Hind et al.). The `conv` prefix
 is proposed here using the convenient LLMON syntax (Chapter 2):

    \conv:user\ what's a good restaurant nearby? /conv:user/
    \conv:intake\ cuisine=?, distance=? /conv:intake/
    \conv:elicit\ What kind of food sounds good? /conv:elicit/
    \conv:user\ Mexican /conv:user/
    \conv:intake\ cuisine=Mexican /conv:intake/
    \reason:exec: recommend, input=cuisine+distance\
      \reason:characterize\ simple recommendation /reason:characterize/
      \reason:execute\ Casa Luna, Beach and Main /reason:execute/
    \conv:assert\ Casa Luna is great — right on Beach and Main. /conv:assert/
    \conv:user\ what do you mean? /conv:user/
    \conv:repair\ It's a Mexican restaurant about two blocks west. /conv:repair/

Reasoning Thoughts (internal, not verbalized) and conversation Thoughts
(verbalized) interleave naturally in a single trace.

### 12.2.4 Constraint Grammar

After a pause, only a Thought tag (exec binding) or terminal (answer) is
valid. Within a Thought span, the model generates freely --- producing
Meaning content. The grammar guarantees Thought/Meaning alternation.

The constraint grammar above covers reasoning traces. A conversation
grammar would need additional productions for the `conv:` tags ---
interleaving user turns, intake parsing, and verbalized Thoughts with
optional reasoning sub-traces. This grammar is not yet specified.

### 12.2.5 Training Data

Mode 0 traces serialize to LLMON training data. The optimization loop
produces traces annotated with credit assignments. Distractor-infused
variants train the model to execute selected Thoughts and ignore
available but unselected ones. Section 12.3 describes the full synthetic
data generation approach.

## 12.3 Synthetic Data Generation

The Meaning/Thought machinery doubles as a synthetic data engine.
Because every reasoning run and every conversation produces a structured
trace of typed (Thought, Meaning\[\], Meaning) entries, and because
Meaning operations can systematically vary those traces, the system
generates training data as a side effect of normal operation --- and can
amplify that data through principled augmentation.

### 12.3.1 What the Traces Contain

A single Mode 0 reasoning run produces:

-   A characterization Meaning (complexity, form, domain)
-   An information inventory (extracted facts as Meanings)
-   A task list (decomposed sub-problems as Meanings)
-   A grounding audit
-   Per-task results (each a Meaning with typed output)
-   Per-task validation records (which IVR requirements passed/failed)
-   A format plan
-   A synthesized answer
-   Credit assignments per thinking pattern (from the optimizer)

A single conversation produces:

-   Intake parse results (NCF action type, slot Meanings, goal signals)
-   Thought selections at each turn (which Thought, why)
-   Verbalized outputs (with the Meaning they realized)
-   Repair sequences (which operation was applied, result)
-   Slot filling trajectories (order, elicitation strategy, user
    responses)

Both trace types are fully structured. Every entry has a Thought name,
typed inputs, typed output, and a verbalized flag. This is not post-hoc
annotation --- it's the execution record.

### 12.3.2 Reasoning Traces

**Direct serialization.** Every reasoning run serializes to LLMON
format: a sequence of (exec binding, phase content, result, pause)
tuples. The optimizer loop generates thousands of these as a side effect
of optimization --- each run on a different prompt with different
thinking pattern selections.

**Distractor-infused variants.** Following the LLMON paper's
methodology: include multiple available thinking patterns in the context
but only one selected via exec binding. The model learns to execute the
selected pattern and ignore the rest. This trains Thought-level
instruction following.

**Credit-annotated traces.** The optimizer's hindsight credit assignment
produces per-Thought quality scores. These serialize as LLMON metadata
(`reason:meta.credit.critique → helpful`). A model trained on these
traces learns not just how to execute Thoughts, but which Thoughts are
likely to be helpful for which kinds of requests.

**Systematic variation via Meaning operations.** Given one good
reasoning trace, Meaning operations generate variants:

    # Original trace from a reasoning run
    trace = reason(m, request, config=config)

    # Vary the request
    for variant in [request.paraphrase(), request.reframe("security"),
                    request.compress(), request.elaborate()]:
        variant_trace = reason(m, variant, config=config)
        corpus.add(variant_trace)

    # Vary thinking pattern selections (exploration)
    for patterns in optimizer.sample_alternatives(trace):
        alt_trace = reason(m, request, config=config.with_patterns(patterns))
        corpus.add(alt_trace, credit=evaluate(alt_trace))

Each variant is a complete, validated trace --- not a noisy
perturbation. The Meaning operations ensure the variants are
semantically coherent.

### 12.3.3 Conversation Traces

**NCF-structured dialogue.** The conversation engine generates traces
where every turn is a Thought with typed Meanings. These serialize as
interleaved `conv:` and `reason:` LLMON spans. Unlike most dialogue
corpora, the traces carry structural annotation for free: which turn is
an elicitation, which is a repair, which is a sequence closer.

**Systematic user simulation.** Because slot Meanings describe what
information is needed (not how to ask for it), the system can generate
synthetic user responses by realizing slot values:

    # Generate a synthetic conversation for find_restaurant
    for cuisine in ["Mexican", "Thai", "Italian"]:
        for distance in ["walking", "driving", "any"]:
            sim = ConversationSimulator(app, domain=restaurant_db)
            sim.inject_slot("cuisine", Meaning(cuisine))
            sim.inject_slot("distance", Meaning(distance))
            trace = sim.run()
            corpus.add(trace)

**Repair path generation.** The NCF defines which repairs are legal
after any agent verbalization. The simulator can exhaustively exercise
them:

    # After every agent turn, try each repair type
    for turn in trace.agent_turns():
        for repair in [turn.output.paraphrase(), turn.output.concretize(),
                       turn.output.repeat(), turn.output.elaborate()]:
            repair_trace = sim.branch_at(turn, user_says="what do you mean?")
            corpus.add(repair_trace)

This produces training data for the long tail of conversational
navigation that real user data rarely covers --- aborts, repairs,
re-initiations, cross-catalog invocations mid-conversation.

### 12.3.4 Bridge Traces

The most valuable traces are ones that interleave conversation and
reasoning. These are hard to collect from real usage but easy to
generate:

    # Conversation that triggers reasoning
    sim = ConversationSimulator(app, domain=restaurant_db)
    sim.inject_slot("cuisine", Meaning("Mexican"))
    # on_complete triggers reason() — the trace captures both
    trace = sim.run()
    # trace contains: conv:elicit → conv:user → reason:exec → ... → conv:assert

    # User repairs a reasoned answer
    sim.continue_with("can you give an example?")
    # trace extends with: conv:repair via concretize() on the reasoned Meaning

These bridge traces teach the model that reasoning Thoughts and
conversation Thoughts share a token stream and that NCF repairs apply to
reasoned answers.

### 12.3.5 Data Quality Controls

**Validation filtering.** Traces where IVR failed (degraded Meanings)
can be excluded from training, or included with lower weight to teach
the model about graceful degradation.

**Credit-weighted sampling.** Traces with higher optimizer credit scores
are sampled more frequently, biasing the training distribution toward
effective Thought selections.

**Deduplication via Meaning similarity.** Traces whose request Meanings
are too similar (by the embedding matcher) can be collapsed to avoid
overfitting to particular phrasings.

**Grounding audit.** Traces where the grounding Thought flagged
ungrounded claims are either repaired before inclusion or excluded. This
prevents training on hallucinated intermediate results.

### 12.3.6 Volume and Cost

A single reasoning run on a moderately complex prompt produces \~15--25
Thought executions at \~500 tokens each. Meaning-operation variants
multiply this cheaply --- paraphrasing a request is one LLM call, but
the resulting trace is 15--25 calls of new content.

Estimated yields:

-   1K prompts × 5 variants × 20 Thoughts × 500 tokens ≈ 50M tokens of
    reasoning traces
-   1K conversation scenarios × 10 slot combinations × 5 repair variants
    ≈ 25M tokens of conversation traces
-   Bridge traces add \~20% on top

The structured format means these traces can be used for SFT (supervised
fine-tuning), DPO (using credit-annotated good/bad trace pairs), or
process reward model training (per-Thought credit scores as step-level
rewards).

### 12.3.7 Thought Grammar

See `thought_grammar.py` for a formal grammar over Thought sequences
that generates valid traces for SDG. The grammar is a weighted CFG with
these productions:

**Reasoning traces.** Fixed pipeline (characterize → extract_info →
decompose → ground → execute\* → format_plan → synthesize) with optional
injection blocks at three points. Injection blocks sample from the
pre-execution, post-execution, and post-synthesis Thought catalogs with
deduplication.

**Conversation traces.** NCF adjacency pair sequences: user turn
(intake) → agent turn (elicit \| assert\_ \| confirm) → optional
repairs. 1--4 exchanges, optionally closed.

**Bridge traces.** Conversation traces where at least one agent turn
expands to a full reasoning sub-trace. The reasoning result is
verbalized via assert\_, and NCF repairs apply to the verbalized output.

**Cross-catalog nesting.** Reasoning can suspend mid-execution to invoke
elicit → intake (ask the user). Conversation can invoke a full reasoning
sub-trace as its agent turn. Nesting depth is configurable.

**Composed Thoughts.** critique.then(revise) is a grammar-level pair:
critique always enables revise, and revise requires a preceding
critique. Other compositions (e.g., abstract\_.then(concretize)) can be
added as paired productions.

The grammar includes seven named profiles from the design doc (default,
high_stakes, quick_advisory, ambiguous_request, architecture_decision,
robust_recommendation, novel_problem) that produce deterministic
sequences matching the documented pattern compositions. The sampler
generates random valid traces with configurable complexity budgets and
LLMON serialization.

## 12.4 Prior Art

### 12.4.1 Rasa CALM

Closest architectural relative for conversation. Parses user input into
structured commands, routes through deterministic Flows. Key
differences: no conversation-analytic grammar, no safety pipeline, no
semantic content type, no connection to reasoning.

### 12.4.2 Prompt Injection Defenses

The security community's control/data plane separation is the same
insight as our Intake Pipeline. Our design goes further: after intake,
user input exists only as Meaning objects placed in data positions. In
Mode 0 this is enforced by Mellea's API (convention); in Modes 1+ it is
enforced structurally by LLMON constraint masking (see Section 1.1).

### 12.4.3 NCF on Watson

The NCF was implemented on IBM Watson Assistant in the IECR paradigm.
Our design transplants it into generative programming. The NCF content
format becomes Meaning operations. Dialog nodes become Thoughts.

### 12.4.4 Factored Cognition / Decomposed Prompting

Ought's Factored Cognition and Khot et al.'s Decomposed Prompting
explored decomposing cognitive tasks into small pieces. Our design adds:
typed content (Meanings), validated operations (Thoughts with IVR),
pluggable optimization, and the conversation bridge.

### 12.4.5 BOAD

Xu et al.'s BOAD formulated agent design as a multi-armed bandit. Our
design applies this to Thought selection with the refinement that
Thoughts have typed Meaning inputs/outputs, enabling richer credit
assignment. BOAD's UCB strategy is the reference optimizer, but the
optimizer interface is pluggable --- alternative learning strategies can
be substituted without changing the Thought/Meaning architecture.

### 12.4.6 Decision-Centric Design

Sun (2025) proposes separating decision-relevant signals from the policy
that maps them to actions, making control explicit and inspectable. The
`(A, c, δ)` abstraction --- actions, decision context, decision function
--- unifies routing, adaptive inference, and sequential act-vs-clarify
decisions. Our design adopts this formalization for Thought selection:
the actions are available Thoughts, the context is accumulated Meanings
plus explicit signals (sufficiency, correctness), and the decision
function is the pluggable optimizer. The paper's attribution framework
(signal error vs. policy error vs. execution error) maps directly onto
our trace structure. Where our design extends this: Thoughts are typed
cognitive operations, not opaque actions, and the decision context is
structured Meanings, not unstructured state.

### 12.4.7 What's Novel

1.  **Meaning/Thought as a unified primitive** for both conversation and
    reasoning. The difference is verbalization, not mechanism.
2.  **Types as Meanings.** The type system is semantic and
    transformable.
3.  **NCF repairs as Meaning operations.** Conversational repair falls
    out of the Meaning abstraction.
4.  **Layered data isolation.** After intake, user input exists only as
    typed Meanings in data positions --- enforced by API convention in
    Mode 0, by token-level constraint masking in Modes 1+.
5.  **Cross-catalog invocation.** Conversation Thoughts can invoke
    reasoning Thoughts and vice versa.
6.  **Decision-centric Thought selection.** Thought selection formalized
    as `(A, c, δ)` with explicit signals, modular policies, and
    three-way failure attribution (signal/policy/execution).

## 12.5 Open Questions

### Meaning

1.  **Realization caching.** When to realize once and cache vs.
    re-realize? Repairs want freshness; confirmations want consistency.

2.  **Type matching failure modes.** The reference implementation uses
    embedding similarity (Section 1.2). Where does cosine similarity
    over type descriptions fail? Types with similar surface form but
    different intent? Types requiring world knowledge to distinguish?

3.  **Meaning serialization.** How are Meanings persisted between
    sessions?

4.  **Meaning composition.** Is `m1.and_(m2)` a single Meaning or a
    MeaningGroup?

5.  **Meaning metadata mutation.** Meanings are immutable
    (`frozen=True`), but should `metadata` support in-place annotation
    (e.g., attaching credit scores post-hoc) via a separate mutable
    sidecar?

### Thought

6.  **Developer-defined Thoughts.** How to register a Thought in the NCF
    grammar or optimizer?

7.  **Thought cost model.** How does the optimizer learn cost-adjusted
    quality?

8.  **Thought composition algebra.** `t.then(u)`, `t.gate(condition)`,
    `t.parallel(u)` --- what's the minimal set?

9.  **Cross-catalog selection.** One optimizer or two when conversation
    and reasoning interleave?

### System

10. **Multi-model routing.** Different models for safety, parsing,
    reasoning, conversation?

11. **Streaming.** `ainstruct()` with Meaning operations and structural
    validation?

12. **Goal detection accuracy.** How conservative for generative goal
    detection?

13. **Mode migration.** How to validate Mode 1 equivalence to Mode 0?

14. **Optimizer interface.** The decision-centric formalization (Section
    2.3) defines the contract: `δ(DecisionContext) → Thought`. Remaining
    questions: what is the minimal `DecisionContext` that supports the
    UCB reference implementation? How should credit assignment propagate
    through composed Thoughts (`a.then(b)` --- does each get separate
    credit)? How does the interface change between Mode 0 (program
    selects) and Mode 2 (model selects)?

### Carried Forward from v0.5

The reasoning scaffold design doc (v0.5, Section 7) raises operational
questions that remain open:

15. **Parallel execution.** Tasks with no mutual dependencies could run
    concurrently. What is the right abstraction for parallel Thought
    execution in Mellea?

16. **Token budgeting.** Can we implement a budget-aware orchestrator
    that dynamically skips Thoughts when approaching a token ceiling?

17. **Compositional interference.** Do some Thoughts interfere? Does
    critique undo constraint propagation? Does counterfactual probing
    after assumption surfacing re-surface the same concerns?

18. **Cross-model transfer.** If the optimizer learns on Granite 8B, do
    the same Thought selections help on Llama 70B? BOAD found partial
    transfer only.

19. **Cost-quality Pareto frontier.** A cost-aware optimizer (reward =
    quality-per-token) would discover optimal Thought compositions at
    each budget level.

20. **LLMON token overhead.** Estimated 15--20% overhead for structural
    tokens; KV cache reuse in single-stream mode may offset this.
    Empirical measurement needed.

21. **Pattern taxonomy validation.** Are there missing categories? Do
    some Thoughts subsume others? The optimizer will reveal this, but
    principled analysis would help.

## 12.6 Implementation Priorities

**Phase 1: Meaning/Thought Core** - `Meaning` class: intent, type (as
Meaning), core operations (paraphrase, negate, compress, elaborate,
abstract, concretize, reframe, challenge, resolve, tone), lazy
evaluation, realization via `m.instruct()` - Soft type matching
(embedding similarity default + LLM judge option, two-threshold
accept/reject model) - `Thought` class: typed I/O (Meanings),
requirements, IVR strategy, injection point, type matcher/strictness -
Trace capture as (Thought, Meaning\[\], Meaning) entries - Thought
composition: `then`, `gate` - Degradation paths: degraded Meanings, IVR
exhaustion fallback, Meaning operation retry

**Phase 2: Conversation** - Intake Pipeline (safety gate, parse to
Meanings, enrich) - Conversation Thoughts: elicit, assert, repair (via
Meaning ops), close, abort - NCF state machine (9-pattern MVP) - Slot
filling with Meaning-based elicitation - Basic goal tracker (declared
goals) - Developer API: `@app.goal`, `@on_complete` - Flow abort
semantics (generator cleanup, slot frame preservation)

**Phase 3: Reasoning** - Core pipeline Thoughts (characterize,
extract_info, decompose, execute, synthesize) - 9 implemented thinking
pattern Thoughts - Optimizer interface + UCB reference implementation
(credit assignment) - `reason()` one-call interface - ThoughtSequence
profiles

**Phase 4: Bridge + Expansion** - Cross-catalog invocation +
SuspendedTrace serialization - P1 reasoning patterns (confidence
calibration, premortem, elimination, contradiction detection) - Process
flows for conversation - Goal discovery from user input - Custom Thought
registration

**Phase 5: LLMON + Advanced** - LLMON trace serialization (reasoning) -
`conv:` tag vocabulary (co-design with LLMON authors) - Mode 1
(single-stream with Thought tags) - Activated LoRA per Thought -
Hypothesis tracking (optional) - Mode 2/3 trajectory

---

> [← Ch.11: Reasoning](11-reasoning.md) · [Table of Contents](../README.md) · [Ch.13: Semantic Operators →](13-semantic-operators.md)
