> [← Ch.12: Unification, Traces, and Synthetic Data](12-unification-traces-sdg.md) · [Table of Contents](../README.md) · [Ch.14: Software-Defined Models →](14-software-defined-models.md)

---

# Chapter 13: Semantic Operators

**Status:** Draft
**Version:** 0.2
**Authors:** Kush R. Varshney, David Cox, Gemini, Claude
**Date Created:** 2026-04-05
**Date Modified:** 2026-04-11

**Prerequisites:** Chapter 8 (Meaning and Thought), Chapter 1 (Span Algebra), Chapter 9 (Decision Architecture)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## From Meanings to Operators

A Meaning (Chapter 8) is a typed semantic object: an output of a Thought with an `intent`, a `type`, a provenance chain, and taint labels. But a Meaning is not inert. It can be transformed, composed, abstracted, and reframed. These transformations are **semantic operators** — typed functions that consume Meanings and produce new Meanings, preserving or modifying their structure.

This chapter formalizes semantic operators: the algebra of transformations over Meanings, their composition laws, and their relationship to Thought execution. Semantic operators implement the **Formulate-Compute-Ground (FCG) pattern**, a design methodology for embedding formal reasoning within generative programs. They are how reasoning and conversation leverage formal compositional structure without sacrificing the flexibility of LLM-driven formulation and grounding.

---

## 13.1 Meaning Operations as Semantic Operators

Semantic operators are Meaning methods that transform one Meaning into another. Unlike formal operators, they have **intentional semantics**: an operation like `m.negate()` produces a Meaning whose realization is a generative computing's best attempt at negation, not a logically guaranteed inversion.

```python
# Semantic operators are methods on Meaning
behavior = Meaning("A researcher prioritizes publishing over reproducibility")
negated = behavior.negate()           # → Meaning with intent "A researcher does not prioritize publishing..."
abstracted = behavior.abstract()       # → Meaning with intent "An academic prioritizes career advancement over integrity"
reframed = behavior.reframe("virtue ethics") # → Meaning "Prioritizing publishing over reproducibility reflects a failure of intellectual honesty"
```

### Fundamental Meaning Operations

Chapter 8 defines the core Meaning operations. Here we explain how they work as semantic operators:

**Intent-preserving transforms** (same core meaning, different expression):
- `m.paraphrase()` — different words, same meaning
- `m.repeat()` — same meaning and detail level
- `m.compress()` — fewer details, same core meaning
- `m.elaborate(context)` — more details, same core meaning
- `m.in_tone(register)` — different register (formal/casual/technical)
- `m.for_audience(perspective)` — adapted for a specific perspective

**Intent-altering transforms** (different meaning):
- `m.negate()` — inverted meaning (but no guarantee `m.negate().negate() ≈ m`)
- `m.abstract()` — move up the abstraction ladder
- `m.concretize(context)` — move down, add specifics
- `m.reframe(perspective)` — same facts, different framing
- `m.incorporate(other)` — merge another Meaning's content
- `m.challenge()` — identify weaknesses or counterarguments
- `m.resolve(issue, info)` — resolve ambiguity with new information

**Composition:**
- `m.and_(other)` — combine two Meanings

This list is not meant to be comprehensive and may grow as more fundamental operators are discovered.

### Bridging Messy Reality and Formal Structure

Formal compositional reasoning has powered the structural turn in linguistics, logic, psychology, and the social sciences. Yet it has remained largely theoretical because it is brittle when applied to messy real-world language and context:

- **Linguistic semantics** (Heim and Kratzer) provides elegant compositional rules for computing meaning via lambda calculus, but struggles with the ambiguity, indexicality, and context-dependence of actual language use.
- **Moral judgment theory** (Wegner and Gray) models how humans assess harm via dyadic agent-patient interactions and multiplicative factors, but real moral judgments depend on mind perception: observers disagree not on the formula but on how strongly they perceive entities as intentional, vulnerable, or causes of harm.
- **Decision architecture** (Chapter 9) formalizes the mapping from decision context (signals and state) to policy (action selection), but real contexts are ambiguous: signals must be estimated from noisy observations, and policies must adapt to incomplete information.
- **Temporal reasoning** (linear temporal logic) provides precise formal operators for reasoning about event sequences and causality, but real temporal relations in natural language are ambiguous: temporal operators must be formulated from context and grounded back to concrete event orderings.
- **Person-environment interaction** (Kurt Lewin's behavior equation `B = f(P, E)`) models how behavior emerges from individual and environment, but real contexts require formulating which characteristics matter.
- **Narrative structure** (Parry and Lord, Lévi-Strauss) reveals compositional patterns in stories, but these patterns are obscured by the messy specifics of language, culture, and individual variation.

Generative models (large language models) excel at handling context, ambiguity, and pragmatic nuance. But they lack the precision, repeatability, and auditability of formal systems.

**The Formulate-Compute-Ground (FCG) pattern bridges this gap.** This pattern decomposes complex reasoning into three phases: pragmatic formulation (disambiguation and structuring via LLM, model, or code), deterministic computation (pure symbol manipulation), and pragmatic grounding (conversion back to natural language and context-dependent output via LLM, model, or code). Together, these phases enable operations that are precise (formal core), adaptive (pragmatic boundaries), and reliable (through validation of the pragmatic outputs).

#### Soft Algebra: Intentional Semantics as Formal Structure Within Pragmatics

Meaning operations have **intentional semantics**, which embeds formal computation *inside* pragmatic contextual processing. For example, `m.negate()` works as follows:

1. **Formulate (pragmatic):** Interpret the Meaning's intent in context. What does it mean to "negate" this particular meaning? What ambiguities exist? The LLM resolves these questions.
2. **Compute (algebraic):** Apply logical negation to the formalized structure.
3. **Ground (pragmatic):** Render the negated structure back into natural language appropriate to the context.

```
m.negate() = formalize (LLM) → computation (logical negation) → ground (LLM)
```

The operation as a whole is pragmatic: the LLM handles context and ambiguity at the boundaries. The middle is deterministic. But you cannot separate them—the operation is the entire three-phase cycle. This structure has important consequences:

- `m.negate()` produces a Meaning whose realization is the result of this three-phase process, not just logical negation.
- There is **no guarantee** that `m.negate().negate()` recovers `m` because pragmatic context may shift during realization (different ambiguity resolutions, different grounding choices). This **pragmatic drift** is a feature, not a bug — it means the operations adapt to context.
- There is **no guarantee** that `m.abstract().concretize()` recovers `m` for the same reason.
- Operations are **not formal functions**, but they contain formal operations inside them.

**Why this is by design:** Formal operations alone are brittle on messy real language. Pragmatic processing alone lacks precision. Intentional semantics wraps formal structure in pragmatic context, making operations reliable through **validation** of outputs against their intended semantics. A Thought that expects a critique validates that the output is actually a critique—whether this validation is implemented via IVR loops, learned classifiers, deterministic checks, or some combination. The Meaning operations are tools for expressing intent; the system's robustness comes from checking whether that intent was achieved.

### Lazy Composition

These operations are **lazy**. They don't produce text immediately. Instead, they accumulate in the Meaning's `_operations` tuple and are realized together in a single LLM call:

```python
chain = behavior.negate().in_tone("formal").concretize()
# chain is a single Meaning with _operations = (negate, in_tone, concretize)
# Realization happens once:
text = chain.realize(session)  # one LLM call produces the full chain
```

When you chain operations, no LLM calls happen until realization. This lazy composition enables efficiency: each operation (negate, in_tone, concretize) can perform its internal Formalize-Compute-Ground cycle.

This lazy composition enables:
- **Efficiency**: fewer LLM calls
- **Composability**: operations chain safely through **soft type matching**
- **Auditability**: the full operation chain is visible before realization

**Soft type matching** (Chapter 8, §8.1.2) is a separate mechanism that applies at Thought boundaries. When a Meaning flows from one Thought to another, the system compares semantic similarity between the Meaning's type and the receiving Thought's expected input type. This is soft (not binary) — a "preliminary finding" is close enough to a "result"; a "greeting" is not. This enables creative composition while preventing category errors.

---


## 13.2 Meaning Operations in Thought Execution

A Thought (Chapter 8, §8.1.3) is a typed cognitive operation that takes Meanings as input and produces Meanings as output. Meaning operations are how Thoughts transform their inputs:

```python
# A Thought that uses Meaning operations
critique = Thought(
    "critique",
    description="Find weaknesses in a result",
    inputs={"target": Meaning("a concrete finding")},
    produces=Meaning("an argument identifying specific weaknesses"),
    requirements=["must identify specific weaknesses, not vague concerns"],
)

# Inside the Thought's execution:
def critique_impl(session, target: Meaning) -> Meaning:
    # Use Meaning operations to prepare the input
    focused = target.concretize()  # add specifics to the finding
    
    # Generate the critique (implementation could be LLM call, model adapter, or other mechanism)
    critique_output = session.generate(
        description="Identify the weaknesses in this finding",
        user_variables={"finding": focused},
        requirements=critique.requirements,
    )
    
    # The output is a new Meaning
    return Meaning(
        intent=critique_output,
        type=critique.produces,
    )
```

Meaning operations compose within Thoughts:

```python
# Prepare input via lazy composition
prepared = target.for_audience("reviewers").elaborate(context=review_guidelines)

# prepared._operations = (for_audience, elaborate)
# Single LLM call produces both transformations
focused = prepared.realize(session)
```

### Soft Type Matching at Thought Boundaries

When a Meaning flows from one Thought to another, type matching is **soft**: the system embeds both the Meaning's type and the receiving Thought's expected type, and compares cosine similarity against a configurable threshold (Chapter 8, §8.1.2).

```python
# "A preliminary finding" ≈ "a concrete finding" (high similarity)
preliminary = Meaning("results lack reproducibility details", type=Meaning("a preliminary finding"))
critique(preliminary)  # accepted with soft match

# "A greeting" ≇ "a concrete finding" (low similarity)
greeting = Meaning("hello", type=Meaning("a conversational greeting"))
critique(greeting)  # rejected: type mismatch
```

This softness is intentional: it allows creative composition while preventing category errors.

---

## 13.3 Semantic Operators in Reasoning and Conversation

Semantic operators implement a core insight from cognitive science: the **Language of Thought Hypothesis** (Fodor, Pylyshyn). This hypothesis proposes that human cognition operates over a compositional, structured internal language—a formal system with type-safe operations and transparent semantics. Semantic operators are this hypothesis made operational in generative computing: they are the typed, compositional transformations that reasoning and conversation use to manipulate structured Meanings.

Semantic operators are the machinery behind two key generative computing patterns: reasoning and conversation.

### Reasoning Pipeline

The reasoning pipeline (Chapter 11) uses semantic operators to refine understanding through multiple steps:

```
Initial Meaning (query or observation)
  ↓ [Characterize via reframe]
  ↓ [Decide if complex reasoning needed]
  ↓ [Abstract to core problem]
  ↓ [Apply domain-specific thinking (e.g., step-by-step reasoning)]
  ↓ [Elaborate with supporting evidence]
  ↓ [Compress for clarity]
  ↓ [Output: Final Meaning]
```

Each arrow is a semantic operator application. The pipeline is structured, not free-form generation. This structure enables:
- **Auditability:** Each step is visible and inspectable
- **Efficiency:** Steps can be skipped if characterization indicates they're unnecessary
- **Compositionality:** Different reasoning strategies can be plugged in via different operator sequences

### Conversation Structure

Conversation (Chapter 10) uses semantic operators to maintain coherence across turns:

1. **User input Meaning** arrives
2. **Reframe** to extract speaker intent, clarify ambiguity
3. **Abstract** to identify topic and relation to prior turns
4. **Compose** with prior conversation state (span algebra composition)
5. **Generate response** Meaning via appropriate reasoning pipeline
6. **Compress** response if necessary to fit token budget
7. **Return** response Meaning

Multi-turn coherence is maintained because each turn's output Meaning carries its provenance. The system can `reframe(current_meaning, context=prior_conversation)` to ensure consistency.

---

## 13.4 Examples of Implementing Semantic Operators

To make this concrete, here are four extended examples of semantic operator implementation. The examples below use `session.instruct()` with IVR validation for illustration, but formulation and grounding could equivalently be implemented via trained adapters, deterministic code, or any combination of LLM and non-LLM mechanisms. The three-phase structure (Formulate-Compute-Ground) remains the same regardless of implementation choice.

### Example 1: Composing Meaning Operations

**Goal:** Show how Meaning operations compose into complex transformations using `and_`. The `and_` operator has many possibilities (logical conjunction, set intersection, temporal sequencing, etc.). Here we focus on the specific case of **coordinating predicates** — combining two simple sentences with shared subjects under a single agent.

**Code Implementation:**

```python
# Two separate Meanings representing related facts
learning = Meaning(
    intent="Kush learns Qiskit",
    type=Meaning("a learning activity")
)

teaching = Meaning(
    intent="Kush teaches Python",
    type=Meaning("a teaching activity")
)

# Combine them using the `and_` operator from Chapter 8
combined = learning.and_(teaching)
# combined._operations = (and_,)
# combined is a new Meaning, not yet realized
```

**Inside the `and_` operation (when realized):**

1. **Formulate (LLM via IVR):**
   ```python
   # Extract semantic roles from both Meanings
   learning_parsed = session.instruct(
       description="Extract subject and predicate",
       user_variables={"statement": "Kush learns Qiskit"},
       requirements=["Return: subject, predicate"]
   )
   # → Meaning: "subject: Kush, predicate: learns Qiskit"
   
   teaching_parsed = session.instruct(
       description="Extract subject and predicate",
       user_variables={"statement": "Kush teaches Python"},
       requirements=["Return: subject, predicate"]
   )
   # → Meaning: "subject: Kush, predicate: teaches Python"
   ```

2. **Compute (formal, deterministic):**
   ```python
   # subject_a, predicate_a, subject_b, predicate_b are available from learning_parsed and teaching_parsed
   
   # Check that subjects match (coordination requires shared agent)
   if subject_a !≈ subject_b:
       raise ValueError(f"Cannot coordinate: subjects differ ({subject_a} vs {subject_b})")
   subject = subject_a
   
   # Lambda abstraction: combine two predicates
   # λx . [A(x)] AND [B(x)]
   # Instantiate: λx . [x learns Qiskit AND teaches Python]
   
   combined_form = f"{subject} {predicate_a} and {predicate_b}"
   # → "Kush learns Qiskit and teaches Python"
   ```

3. **Ground (LLM via IVR):**
   ```python
   # Ensure the combined statement is natural and coherent
   grounded = session.instruct(
       description="Rephrase this naturally",
       user_variables={"statement": combined_form},  # from Compute step
       requirements=["Ensure grammatical elegance"],
       ivr_strategy="validate"
   )
   # → Meaning: "Kush learns Qiskit and teaches Python"
   ```

**Why this exemplifies intentional semantics:**

The `and_` operation is **not purely formal** (it needs LLM judgment to extract roles and ensure natural grounding) and **not purely pragmatic** (it uses formal lambda structure to correctly combine predicates under the same subject). Instead:
- The formal core (λx . A(x) AND B(x)) is deterministic: given the same parsed roles, always produces the same logical form
- The pragmatic wrapper (formulation and grounding) handles ambiguity in role extraction and natural language rendering
- Validation ensures the combined Meaning is actually coherent

This shows how **Meaning operations embed formal structure inside pragmatic processing**. The developer can reason about the lambda structure (ensuring roles align, predicates combine correctly), but the operation's reliability comes from pragmatic validation, not formal guarantees.

### Example 2: Voice Transformation and Grammatical Reframing

**Goal:** Show how Meaning operations handle grammatical transformations—transforming active voice to passive voice while preserving semantic content through lambda abstraction.

**Code Implementation:**

```python
# A Meaning in active voice
active = Meaning(
    intent="Kush learns Qiskit",
    type=Meaning("a statement in active voice")
)

# Transform to passive voice using the `reframe` operation
passive = active.reframe("passive_voice")
# passive._operations = (reframe("passive_voice"),)
# passive is a new Meaning, not yet realized
```

**Inside the `reframe` operation (when realized):**

1. **Formulate (LLM via IVR):**
   ```python
   # Parse the active voice sentence into semantic roles
   parsed = session.instruct(
       description="Parse active voice sentence into subject, verb, and object",
       user_variables={"sentence": "Kush learns Qiskit"},
       requirements=[
           "Return structured: subject (doer), verb, object (affected)",
       ]
   )
   # → Meaning: "subject: Kush, verb: learns, object: Qiskit"
   ```

2. **Compute (formal, deterministic):**
   ```python
   # subject, verb, obj are available from parsed
   
   # Active voice: λx . x verb object
   # active_lambda = lambda x: f"{x} {verb} {obj}"
   # active_lambda(subject) → "Kush learns Qiskit"
   
   # For passive: reframe the predicate
   # The Passive Predicate (the new 'property' of the object)
   passive_predicate = f"is {verb}ed by {subject}"
   # Example: "is learned by Kush"
   
   # The New Lambda (waiting for the new subject: the object)
   passive_lambda = lambda y: f"{y} {passive_predicate}"
   
   # Apply the lambda to the new subject (the object, now topical)
   passive_sentence = passive_lambda(obj)
   # → "Qiskit is learned by Kush"
   ```

3. **Ground (LLM via IVR):**
   ```python
   # Ensure the passive form is natural English
   grounded = session.instruct(
       description="Ensure natural passive voice phrasing",
       user_variables={"passive_sentence": passive_sentence},
       requirements=[
           "Validate: is the passive form grammatical?",
           "Ensure: the semantic meaning is preserved from active form",
           "Check: irregular verb forms are correct (e.g., 'taught' not 'teached')"
       ],
       ivr_strategy="validate"
   )
   # → Meaning: "Qiskit is learned by Kush" (or corrected form if needed)
   ```

**Why this exemplifies intentional semantics:**

The `reframe("passive_voice")` operation is neither purely formal nor purely pragmatic:
- The formal core (lambda abstraction for subject and predicate repositioning) is deterministic: given the same parsed roles, always produces the same logical form
- The pragmatic wrapper (LLM judgment in Formulate and Ground) handles role identification, irregular verb morphology, and validates that natural English phrasing is preserved
- The semantic intent remains invariant across transformation—both active and passive refer to the same event (Kush, learning, Qiskit), just with different narrative perspective and topic structure

This demonstrates that **grammatical transformations are not string operations but semantic operations where the lambda structure preserves what changes (subject position and verb form) while ensuring semantic equivalence through validation**.

### Example 3: Abstraction

**Goal:** Abstract a specific fact to a general one: "Kush learns Qiskit" → "Researchers learn Qiskit."

**Code Implementation:**

```python
# A concrete Meaning
specific = Meaning(
    intent="Kush learns Qiskit",
    type=Meaning("a concrete learning activity")
)

# Chain operations lazily (no LLM calls yet)
abstracted = specific.abstract()
# abstracted._operations = (abstract,)
# text = abstracted.realize(session)  # Would happen later
```

**Inside the abstraction, three phases execute:**

**1. Formulate (LLM via instruct):**

```python
# Session formulates the semantic structure
formulated = session.instruct(
    description="Extract the subject and predicate from this sentence",
    user_variables={"sentence": "Kush learns Qiskit"},
    requirements=["Return: subject, predicate"]
)
# formulated → Meaning with intent "subject: Kush, predicate: learns Qiskit"
```

**2. Compute (formal, deterministic):**

```python
# predicate is available from formulated

# Pure lambda calculus: no LLM
# λx . x learns Qiskit
quantifier = lambda domain_element: f"{domain_element} {predicate}"

# This is the formal core of abstraction
# Apply the quantifier to a generalized domain (computed in step 3)
```

**3. Ground (LLM via IVR):**

```python
# subject is available from formulated

# Pragmatically determine what domain generalizes the subject
generalized = session.instruct(
    description="What category generalizes this person in the context of this activity?",
    user_variables={"subject": subject, "predicate": predicate},
    requirements=["Return a 1-2 word category like 'Researchers' or 'Students'"],
    ivr_strategy="validate"
)
# generalized → Meaning with intent "IBM Researchers" (LLM-determined category)

# domain_category is available from generalized

# Apply the lambda quantifier to the generalized domain
result = quantifier(domain_category)  
# result → "IBM Researchers learns Qiskit"

# Final validation via IVR: ensure result is natural and preserves intent
final = session.instruct(
    description="Rephrase this naturally if needed",
    user_variables={"statement": result, "original_predicate": predicate},
    requirements=[
        "Ensure: the generalization is accurate and sounds natural",
        "Preserve the predicate meaning from the original"
    ],
    ivr_strategy="validate"
)
# → Meaning: "IBM Researchers learn Qiskit"
```

**Why this exemplifies intentional semantics:** The lambda calculus (step 2) is deterministic and formal. But steps 1 and 3 (pragmatic formulation and grounding) use LLM judgment. The operation's reliability comes from IVR validation: if the Thought that consumes this Meaning expects "a general learning activity," IVR will verify the output matches that type.

### Example 4: Reframing for Moral Judgment

**Goal:** Evaluate the moral status of "Doctor euthanizes patient" under a given ethical perspective using the `reframe` semantic operator based on dyadic morality theory.

**Code Implementation:**

```python
# An event Meaning
event = Meaning(
    intent="Doctor euthanizes patient",
    type=Meaning("a morally significant event")
)

# Reframe to different ethical perspectives
buddhist_view = event.reframe("Buddhist ethics: compassion and suffering")
utilitarian_view = event.reframe("Utilitarian ethics: maximizing well-being")
kantian_view = event.reframe("Kantian deontological ethics: universal duties")
```

**Inside reframing, three phases execute (with perspective as a parameter):**

```python
# The perspective is passed in as a parameter
perspective = "Buddhist ethics: compassion and suffering"
```

**1. Formulate (LLM via instruct):**

```python
# Extract semantic roles: agent, patient, action
# Different perspectives perceive these roles differently
perspective_roles = session.instruct(
    description="Under the given ethical framework, analyze the intentionality of the agent, vulnerability of the patient, and damage of the action.",
    user_variables={
        "event": event.intent,  # "Doctor euthanizes patient"
        "framework": perspective
    },
    requirements=["Return: intentionality (0-1), vulnerability (0-1), damage (0-1)"]
)
# → Meaning: "intentionality: 0.3, vulnerability: 0.9, damage: 0.4"
#   (Buddhist view: doctor may have compassionate intent, patient can suffer, but harm is mild)
```

**2. Compute (formal, deterministic):**

```python
# Extract numerical factors from the formulated Meaning
# perspective_roles.intent contains: "intentionality: 0.3, vulnerability: 0.9, damage: 0.4"
# (extraction details omitted; assume factors are available)
# intentionality = 0.3, vulnerability = 0.9, and damage = 0.4 are available from perspective_roles

# Pure dyadic morality formula: no LLM
moral_judgment_score = intentionality * vulnerability * damage
# = 0.108 (deterministic: given same parsed factors, always same result)
```

**3. Ground (LLM via IVR):**

```python
# Validate and express judgment in perspective-appropriate language
judgment_meaning = session.instruct(
    description="Express this moral judgment in terms appropriate to the given ethical framework",
    user_variables={
        "score": moral_judgment_score,
        "framework": perspective,
        "event": event.intent
    },
    requirements=["Explain reasoning grounded in the framework's core values"],
    ivr_strategy="validate"  # IVR ensures output is coherent moral reasoning
)
# → Meaning: "From a Buddhist perspective (score: 0.108), this act reflects low intentionality but acknowledges the patient's vulnerability. The low overall judgment suggests the compassionate intent mitigates the moral weight, despite the suffering involved."
```

**Perspective-based results:**

| Event | Buddhist | Kantian | Utilitarian | Care Ethics | Virtue Ethics |
|-------|----------|---------|-------------|-------------|---------------|
| Doctor euthanizes patient | 0.1080 | 0.2760 | 0.0425 | 0.2550 | 0.1165 |

(Buddhist score computed in the example above; other scores from parallel runs with Kantian, Utilitarian, Care Ethics, and Virtue Ethics perspectives.)

**Why this exemplifies intentional semantics:**

- **Formulate varies:** Different perspectives (LLM judgment) perceive agent intentionality, patient vulnerability, and damage severity differently
- **Compute is fixed:** The dyadic multiplication formula is identical for all perspectives (deterministic, formal)
- **Ground validates:** Validation ensures the output Meaning explains the perspective's ethical reasoning, not just numerical output

The same event produces different moral judgments because perspective-based reframing changes the Formulate phase. This shows how **disagreements about morality are traceable to specific differences in mind perception** (the Formulate step) rather than opaque intuition.

## 13.5 Discussion and Open Questions

### FCG as a Meta-Pattern: Embedding Formal Structure in Pragmatic Context

The FCG pattern is an instance of a deeper principle running through all of computer science: **embedding formal, compositional structure within pragmatic, contextual processing**. This is not a limitation or workaround—it is the fundamental structure that enables systems to be both precise and adaptive.

**Boolean algebra** embeds formal computation in physical constraints: discrete symbols computed via deterministic truth tables, grounded back to voltage levels and empirical circuit behavior.

**Span algebra** embeds formal composition in the model's internal structure: KV cache interdependencies made explicit as named, scoped regions. A span's scope describes which upstream tokens participated in its generation. This explicit structure enables understanding and composing the model's representations.

**Meaning operations** embed formal semantics in pragmatic context: formal operations (negation, quantification, etc.) computed *inside* a Formalize-Compute-Ground cycle, where Formalize and Ground are LLM-driven pragmatic processing.

The pragmatic layers (formalize and ground) handle context, ambiguity, and contingency. The formal layer guarantees compositionality and precision. Together, they enable operations that are:

- **Precise:** The formal core is deterministic and auditable
- **Adaptive:** The pragmatic wrapper handles real-world messiness
- **Reliable:** Validation checks that the operation succeeded in achieving its intent

**Understanding generative computing is understanding intentional semantics:** formal structure enables precision, pragmatic processing enables adaptability, and their integration enables reliability.

### FCG Across Multiple Scientific and Cognitive Domains

The FCG pattern is not limited to semantic operators. It appears in multiple domains studied in linguistics, cognitive science, psychology, and the social sciences. These domains provide the theoretical foundation for FCG within generative computing.

#### Formal Semantics: Heim and Kratzer

**The problem:** How does a listener extract meaning from a sentence's syntactic structure?

**The FCG structure:**

- **formulate:** Map each word and phrase to a semantic type (individual `e`, truth value `t`, predicate `⟨e, t⟩`, quantifier `⟨⟨e, t⟩, t⟩`, etc.). Resolve pronouns and indexicals. This phase handles the ambiguity and context-dependence of natural language.

- **compute:** Apply function types via lambda calculus. A transitive verb of type `⟨e, ⟨e, t⟩⟩` combines with an object of type `e` to produce a predicate of type `⟨e, t⟩`. The predicate combines with a subject of type `e` to produce a truth value `t`. This computation is deterministic—given the same syntactic structure and type assignments, always the same logical form.

- **ground:** Convert the logical result (a truth value or abstract proposition) back into pragmatic meaning: a statement, denial, question, or speech act adapted to context and audience.

#### Moral Judgment: Wegner and Gray's Dyadic Model

**The problem:** How do humans judge the morality of events? Why do people disagree about moral judgments?

**The FCG structure:**

- **formulate:** Identify to what degree entities in an event are **agents** (intentional actors), **patients** (recipients of action who can suffer), and **causes of harm**. This evaluation depends on mind perception: one observer sees a corporation as an intentional agent; another sees it as a legal abstraction with no agency.

- **compute:** Apply a deterministic dyadic formula: 
  ```
  moral_judgment = agent_intentionality × patient_vulnerability × cause_damage
  ```
  Where each factor ranges from 0 to 1. This computation is deterministic: given the same mind-perception assignments, always the same moral score.

- **ground:** Express the judgment as outrage, moral condemnation, demand for restitution, policy action, legal precedent, or teaching opportunity. The same formal judgment can ground differently depending on context.

**Why FCG matters here:** Disagreements about moral judgments are often attributed to conflicting intuitions. FCG makes them traceable: Did observers disagree in the formalize phase (different mind perception) or the formal computation phase (should never happen—the math is fixed) or the ground phase (different pragmatic expression)? Most moral disagreements stem from phase 1.

#### Decision Architecture: Formal Action Selection

**The problem:** Given candidate actions and a decision context (current state, budget, signals), which action should an agent execute?

**The FCG structure:**

- **formulate:** Construct a **DecisionContext** from noisy signals, accumulated observations, and empirical outcomes. Extract and estimate what matters from raw information.

- **compute:** Apply a formal decision policy to select an action. Policies can be rule-based, optimization-based, learned, or LLM-based.

- **ground:** Translate the selected abstract action into concrete execution within the deployment environment. "Invoke tool X" may ground to an API call in cloud, a local function in edge, or cached results in offline.

**Why FCG matters here:** Formal decision policies are powerful but brittle without formalize and ground. Formulation ensures signals are interpreted consistently. Grounding ensures decisions translate correctly to pragmatic action. Chapter 9 formalizes this layer.

#### Temporal Reasoning: Linear Temporal Logic

**The problem:** How do we reason about the ordering and causality of events over time?

**The FCG structure:**

- **formulate:** Interpret natural language descriptions of temporal sequences, events, and constraints. Extract predicates and temporal relations from noisy or ambiguous input.

- **compute:** Apply formal temporal logic operators: Next (X), Always (G), Eventually (F), Until (U). Compose temporal propositions deterministically. Given the same temporal formulation, reasoning about sequences is precise and auditable.

- **ground:** Translate formal temporal specifications back to concrete event traces, execution plans, or system behaviors. The same temporal constraint grounds differently in scheduling domains, verification problems, or narrative sequencing.

#### Person-environment interaction: Kurt Lewin's Behavior Equation

**The problem:** What determines human behavior?

**The FCG structure:**

- **formulate:** Characterize the person (P) and the environment (E). This requires judgment: what aspects matter?

- **compute:** Apply Lewin's equation `B = f(P, E)`. Given P and E, compute B deterministically.

- **ground:** Interpret the predicted behavior in context. "Aggressive behavior" grounds differently in a competitive sport vs. a classroom.

#### Narrative Structure: Parry, Lord, and Lévi-Strauss

**The problem:** Why are stories recognizable and retellable across cultures and centuries?

**The FCG structure:**

- **formulate:** Extract narrative roles and relations: hero, villain, quest, obstacle, resolution, reward/punishment.

- **compute:** Apply compositional rules for how narrative elements combine (studied by Parry, Lord, and Lévi-Strauss).

- **ground:** Render the formal narrative structure in specific language, cultural context, and medium. The same narrative skeleton grounds differently as a written novel, an oral epic, a screenplay, or a song.

#### Unifying Principle: Why These Domains Matter for FCG

All of these domains follow the same structure: **messy formulation → deterministic computation → context-dependent grounding**. They show that:

1. **Formal compositional reasoning is powerful** but only works within abstracted, structured domains
2. **Messiness enters at the boundaries** (formulation from reality, grounding back to reality)
3. **The middle can be deterministic and auditable** if formulation is done well
4. **Different observers/contexts can agree on the computation but diverge in formulation or grounding**, making disagreements traceable

These insights, drawn from decades of research in linguistics, psychology, and cognitive science, are what make FCG effective as a design pattern for generative computing.

### Open Questions

A couple of aspects of the semantic operator model remain unresolved:

**Operator optimization.** Given a goal output type and input type, what is the optimal sequence of operators? This is analogous to database query optimization, but for semantic transformations. A compiler that auto-generates efficient operator sequences would be valuable.

**Operator learning.** Can we learn semantic operators from examples? Given pairs of Meanings representing "before abstraction" and "after abstraction," can we train a function that generalizes? This would enable domain-specific operator suites.

---

## 13.6 Related Work: Neuro-Symbolic FCG Architectures

The **Formulate-Compute-Ground (FCG)** pattern described in this chapter aligns with a growing body of research that seeks to augment LLMs with symbolic reasoning capabilities. Two primary paradigms—**deterministic logical solvers** and **probabilistic world models**—provide external validation for the FCG approach.

### 13.6.1 Deterministic Logical Reasoning: LOGIC-LM

**LOGIC-LM** (Pan et al., 2023) addresses the "unfaithful reasoning" problem in LLMs by separating linguistic processing from logical deduction. This architecture maps directly onto the FCG phases:

- **formulate:** The LLM translates a natural language problem into a formal symbolic language, such as **First-Order Logic (FOL)** or **Answer Set Programming (ASP)**.

- **compute:** A deterministic symbolic solver (e.g., **Z3** or **Clingo**) executes the logical code, ensuring the reasoning is mathematically verified and free from hallucinations.

- **ground:** The LLM interprets the solver's technical output back into a human-readable explanation.

This framework demonstrates that for high-stakes logical deduction, the symbolic "middle" must be deterministic to remain auditable and precise.

### 13.6.2 Probabilistic World Modeling: Rational Meaning Construction

While Logic-LM focuses on rigid deduction, **"From Word Models to World Models"** (Wong, Grand, et al., 2023) explores FCG as a mechanism for building mental simulations of reality. This approach utilizes a **Probabilistic Language of Thought (PLoT)**:

- **formulate:** The LLM acts as a translator, mapping context-sensitive language into structured probabilistic programs (e.g., **WebPPL**).

- **compute:** Instead of a binary solver, the system uses a reasoning engine to perform **Bayesian inference**, allowing the model to represent uncertainty and cause-and-effect.

- **ground:** The resulting world model is used to predict physical interactions, social intentions, or visual scenes, effectively grounding abstract code in a simulation of reality.

### 13.6.3 Synthesis with Semantic Operators

These works confirm the utility of embedding formal structure within pragmatic boundaries. Where **Logic-LM** uses FCG for **verification** and **Wong et al.** use it for **representation**, the semantic operators in this chapter provide a unified **algebra of transformation**. 

By treating operations like `m.negate()` or `m.reframe()` as FCG cycles, we enable generative programs to manipulate structured Meanings with the same rigor found in these specialized neuro-symbolic architectures. This structure ensures that operations remain precise through their formal core while staying adaptive through their pragmatic wrappers.

> [← Ch.12: Unification, Traces, and Synthetic Data](12-unification-traces-sdg.md) · [Table of Contents](../README.md) · [Ch.14: Software-Defined Models →](14-software-defined-models.md)
