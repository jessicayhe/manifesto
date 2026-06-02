> [← Ch.6: Span Attention](06-span-attention.md) · [Table of Contents](../README.md) · [Ch.8: Meaning and Thought →](08-meaning-and-thought.md)

---

# Chapter 7: Mellea Standard Library

**Component:** `mellea.stdlib`
**Status:** Draft
**Version:** 0.5
**Authors:** David Cox, Claude
**Date Created:** 2026-04-04
**Date Modified:** 2026-04-06

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## Patterns for Generative Programming

The preceding chapters describe infrastructure: the span algebra (Chapter 1), LLMON (Chapter 2), model-runtime codesign (Chapter 3), and the `mellea-core` systems that implement them (Chapters 4–6). This chapter introduces **`mellea.stdlib`** — the developer-facing library that makes generative computing practical. `mellea.stdlib` provides the programming patterns, abstractions, and software engineering conventions that let developers write reliable generative programs without constructing spans, managing scopes, or assembling prompts by hand.

The relationship between `mellea-core` and `mellea.stdlib` is the relationship between a kernel and its standard library. `mellea-core` provides mechanisms (span lifecycle, scope enforcement, KV pool management, boundary-constrained masking). `mellea.stdlib` provides policies and patterns (how to structure a generative call, how to validate its output, how to compose generative and deterministic code). A developer using `mellea.stdlib` never touches a span directly — the library assembles spans, scopes, and LLMON structure on the developer's behalf.

## 7.1 The Encapsulation Principle

The organizing principle of `mellea.stdlib` — and of generative computing broadly — is that **it should not matter to the caller how a generative function is implemented**.

When a developer calls `summarize(passage)` or `detect_hallucinations(output)`, they should receive a result. Whether that result was produced by a natural-language instruction assembled into a prompt, by a LoRA adapter that rewires the model's weights, by a behavior baked into the model during training, or by some future mechanism not yet conceived, is an implementation detail hidden behind the function's interface. The standard encapsulation principle from software engineering, applied to generative computation.

This has a direct consequence. LLMs have a "native syntax" — natural language — through which they are normally instructed. Most generative functions work by telling the model what to do in this native syntax. But there is no reason every function must be implemented this way. **LLM intrinsic functions** (or simply intrinsics) are function-like behaviors that bypass the natural-language instruction interface entirely, producing results through mechanisms that modulate the model's computation in ways that cannot be expressed as prompts. Named by analogy to compiler intrinsic functions (which bypass regular language syntax to access processor-level features), intrinsics can be implemented at multiple levels of the stack:

**Scope-injection intrinsics.** The runtime intercepts a structured signal and injects additional instructions into the scope. This works through the model's native language interface, but the injected instructions are composed by the runtime, not the developer.

**Activated LoRA intrinsics.** A LoRA is dynamically applied to model weights within the scope of a specific span, while original weights govern all other spans. The LoRA can induce behaviors radically different from the base model — not by instructing differently, but by temporarily making it a different model.

**Baked-in intrinsics.** The model is trained to recognize structured signals in its input (e.g., a `<detect_hallucinations/>` tag) and produce qualitatively different output in response — capabilities that extend the model's behavioral repertoire beyond what any prompt could elicit.

The encapsulation principle is what makes intrinsics powerful, not merely exotic. Because the calling interface is the same regardless of implementation, a function can be **progressively upgraded** without changing any calling code. An initial implementation might use scope-injection; a later version might be replaced by an activated LoRA (faster, more reliable); a future model release might bake the behavior in natively (fastest, most robust). The developer's program does not change.

## 7.2 The Session

`mellea.stdlib`'s central abstraction is the **MelleaSession**, which corresponds to the generative computing runtime described in Chapter 3. A session manages the inference backend, the context history, and the lifecycle of generative components:

```python
import mellea
m = mellea.start_session()
result = m.chat("What is the etymology of mellea?")
```

The session encapsulates the backend adapter, input/output processors, and controller. Backend-specific concerns (tokenization, special token handling, model-specific prompting) are confined to the adapter. The developer interacts with a model-independent API.

## 7.3 Instructions and Requirements

The most basic `mellea.stdlib` component is the **instruction**: a request to the LLM, accompanied by requirements that the output must satisfy. This realizes the **instruct-validate-repair (IVR)** pattern — the fundamental rhythm of reliable generative programming:

```python
from mellea.stdlib.sampling import RejectionSamplingStrategy

email = m.instruct(
    "Write an email to invite all interns to the office party.",
    requirements=["Be formal.", "Use 'Dear interns' as the greeting."],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)
```

In span-algebra terms, each IVR iteration is a new `gen()` call. Requirements serve three distinct roles, each mapping to a different span-level operation:

**During generation:** requirements are included in the instruction's scope, influencing the model's output.

**During validation:** each requirement is evaluated against the generated output — by an LLM-as-judge, a deterministic Python function, or a specialized verifier model.

**During repair:** if validation fails, the system re-generates with the failed requirement given additional emphasis.

Requirements can be natural-language strings (validated by LLM judge), Python functions returning a boolean, or **checks** — requirements used only for validation, excluded from the generation prompt. Checks are useful for avoiding the "do not think about X" priming effect, where including a prohibition in the prompt paradoxically increases the likelihood of the prohibited behavior.

**Sampling strategies** are pluggable: rejection sampling (generate until requirements pass or budget exhausted), majority voting (generate multiple candidates, select by consensus), and extensible strategies for inference-time scaling. Switching strategies requires changing a single parameter, not rewriting the program.

## 7.4 Generative Slots

A **generative slot** is a function whose implementation is provided by an LLM rather than by conventional code:

```python
from typing import Literal
from pydantic import BaseModel
from mellea import generative

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    score: int  # 1-5
    summary: str  # one sentence

@generative
def analyze_review(text: str) -> ReviewAnalysis:
    """Extract sentiment, a 1-5 score, and a one-sentence summary."""
    ...

result = analyze_review(m, text="Battery life is great but the screen is dim")
```

The function signature specifies the interface: the docstring becomes the prompt, type annotations become the output schema, and Pydantic models define structured output types. The model's output is constrained to conform to the declared types.

This is the span algebra's `gen([I, P'])` pattern — an instruction span and a data span — encapsulated as a Python function with typed inputs and outputs. The developer never constructs spans, scopes, or tags directly; the runtime assembles them from the function signature.

**Constrained decoding** and **requirements validation** are complementary mechanisms. Constrained decoding (for backends that support it) enforces output structure at the token level during generation — guaranteeing structural conformance (valid JSON matching the schema). Requirements validation checks semantic properties (the content is formal, the greeting is correct). The two compose: constrained decoding ensures the output parses; requirements ensure it is correct.

## 7.5 MObjects: The Bridge Between Traditional and Generative Computing

Object-oriented programming groups related data and operations into classes. **MObjects** bring this pattern to generative programs: the `@mify` decorator transforms any Python class into an object that an LLM can interact with, exposing selected fields as context and selected methods as tools:

```python
from mellea.stdlib.mify import mify

@mify(fields_include={"table"}, template="{{ table }}")
class CompanyDatabase:
    table: str = """| Store | Sales |
    | Northeast | $250 |
    | Southeast | $80 |
    | Midwest | $420 |"""

    def transpose(self):
        # classical Python method
        ...

m = mellea.start_session()
db = CompanyDatabase()
answer = m.query(db, "What were sales for the Northeast branch?")
```

In span-algebra terms (Chapter 1, §1.4), the MObject pattern places data spans (the object's fields) alongside instruction spans (its methods) in a unified scope, using fan-in composition so that fields do not cross-taint each other. The `@mify` decorator controls which fields and methods are visible — selective scoping applied at the Python API level.

The bridge is **bidirectional and live**:

**Mirroring.** Objects in the calling Python scope are visible to the model without manual prompt construction. If a field is updated in Python between generative calls, the model sees the updated value.

**Manipulation.** The model can request data updates (setting fields, appending to logs), method calls (dispatched to the Python object, results returned as new spans in the model's context), and restructuring (reorganizing data, splitting or merging objects — validated against type constraints by the runtime).

**Creation.** The model can request construction of new Python objects. When a generative program's output includes structured data matching a known type (Pydantic model, dataclass, MObject class), the runtime instantiates the corresponding Python object automatically. The model is not merely a text generator producing strings for the developer to parse — it is a participant in the program's object graph.

As discussed in Chapter 2 (§2.1), this is not pure serialization. The span algebra's fan-in structure means that structured objects remain structured in their LLM representations. The consequence is a notion of passing by both **value** (content materialized into a span) and **reference** (span points to getter/setter machinery managed by the runtime). Methods on the object are exposed as callable spans — akin to tool calling, but collocated with the object's data, enabling fine-grained interaction mediated by the runtime.

## 7.6 Context Management

Generative programs accumulate context as they execute. `mellea.stdlib` manages this through two mechanisms:

**CBlocks (cached blocks).** Content reused across multiple generative calls is prefilled once and cached — the span algebra's `fill()` with reuse:

```python
from mellea.stdlib import CBlock

reference = CBlock(open("large_reference_doc.txt").read(), cache=True)
for query in queries:
    result = m.instruct(
        "Using the reference document, answer: {{query}}",
        user_variables={"query": query},
        grounding_context={"reference": reference},
    )
```

The reference block is prefilled once; each subsequent query pays only for its own suffix tokens. Under NoPE (Granite hybrid), the cached KV is valid at any composition position — different queries compose with the same cached block without re-prefill.

**Session-level context.** A Context object stores a (possibly partial) history of prior interactions. Context management strategies — truncation, summarization, selective inclusion — let the developer control how history accrues without manually constructing token sequences.

## 7.7 Backend-Agnostic Design

`mellea.stdlib` works across inference backends: Ollama, vLLM, HuggingFace Transformers, OpenAI, Watsonx, LiteLLM, Bedrock, and `mellea-core`. The same `m.instruct()` call, the same `@generative` function, the same `@mify` class works whether the underlying model runs locally or in the cloud.

This is the abstraction boundary from Chapter 3 (§3.3) made concrete. At the top end, `mellea-core` with a Granite NoPE model provides full structural enforcement — boundary-constrained masking, free-monoid composition, taint tracking. At the bottom end, an OpenAI API provides prompt-assembly approximation — the same `mellea.stdlib` program, the same results (approximately), without structural guarantees.

Switching backends requires changing a single parameter, not rewriting the program. This is the generative computing analog of recompiling for a different processor architecture: the same source, different capabilities. A team can deploy with prompt assembly today and swap in `mellea-core` tomorrow, gaining structural guarantees without touching application code.

## 7.8 Circumscribing Nondeterminism

The core insight of generative programming is that stochastic operations must be **circumscribed** by requirement verification. Each generative call is a point of nondeterminism; each requirement check is a point of deterministic control. The IVR loop is the fundamental rhythm — `try/catch` for generative programs.

`mellea.stdlib` makes this rhythm explicit and composable. Instructions carry requirements that are validated after each generation attempt. Generative slots carry type annotations enforced via constrained decoding or schema validation. MObject methods carry type constraints that the runtime validates on every model-initiated update. Thoughts (Chapter 8) carry requirements and soft type matching on their inputs and outputs.

At every level of the stack, the pattern is the same: generate, validate, repair if needed, capture the result as a typed object. The nondeterminism of the LLM is bounded by the determinism of the validation.

## 7.9 From `mellea.stdlib` to the Semantic Layer

The constructs described in this chapter — instructions, generative slots, MObjects — are the building blocks. They operate on text: instructions are strings, requirements are strings, generative slot outputs are parsed from strings. The model produces tokens, and the runtime interprets them.

Chapters 8–11 describe a higher level of abstraction: the **Meaning/Thought** framework, where the system operates on semantic content (Meanings) rather than text, and cognitive operations (Thoughts) transform Meanings into Meanings. Meanings serialize via LLMON (Chapter 2). Thoughts execute via `m.instruct()` with requirements — the same IVR machinery described in §7.3. The encapsulation principle (§7.1) applies: a Thought can be backed by a prompted instruction, an activated LoRA, or a baked-in intrinsic.

The semantic layer does not replace `mellea.stdlib` — it builds on it. `mellea.stdlib` provides the mechanisms; the Meaning/Thought framework provides the programming model for systems that reason and converse.

---

> [← Ch.6: Span Attention](06-span-attention.md) · [Table of Contents](../README.md) · [Ch.8: Meaning and Thought →](08-meaning-and-thought.md)
