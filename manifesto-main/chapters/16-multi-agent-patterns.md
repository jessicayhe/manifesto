> [← Ch.15: Security Across the Stack](15-security.md) · [Table of Contents](../README.md) · [Ch.17: Frontiers →](17-frontiers.md)

---

# Chapter 16: Multi-Agent Patterns

**Status:** Draft
**Version:** 0.1
**Authors:** Ismael Faro, Claude
**Date Created:** 2026-04-06
**Date Modified:** 2026-04-06

**Prerequisites:** Chapter 8 (Meaning and Thought), Chapter 1 (Span Algebra), Chapter 7 (Mellea Standard Library)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## 16.0 From Single-Agent to Multi-Agent

The preceding chapters address a single generative agent operating within a structured context: one model, one scope, one reasoning trace at a time. The 2026 market has moved decisively beyond this. LangGraph, CrewAI, AutoGen, Google's Agent-to-Agent protocol (A2A), and Anthropic's multi-agent orchestration patterns all treat multi-agent coordination as a first-class concern. Teams of specialized agents — planners, coders, reviewers, tool-callers — collaborate on tasks that exceed any single agent's capacity or context window.

This chapter shows that multi-agent orchestration is not a new primitive in the generative computing stack. It is a **composition pattern** over existing primitives. The span algebra (Chapter 1) already provides agent isolation through scopes. Meanings (Chapter 8) already provide typed inter-agent communication. Taint tracking (Chapter 15) already prevents cross-agent information leakage. The infrastructure is in place; what remains is to describe the patterns.

The structural advantage is significant. In conventional multi-agent frameworks, agents communicate by passing strings — natural-language messages that carry no type information, no provenance, and no structural guarantees about what is an instruction and what is data. The generative computing stack provides all three. An agent's output is a Meaning with a type and a provenance record. The receiving agent's intake pipeline (Chapter 10) parses that Meaning into its data plane, where it cannot be confused with instructions. Taint labels propagate across agent boundaries without additional machinery. These are not features bolted onto a multi-agent framework after the fact; they are consequences of the architecture described in Chapters 1 through 15.

## 16.1 Agent Isolation via Scopes

An agent, in the generative computing model, is a Thought (Chapter 8) executing within its own scope. Agent A's context — its instructions, its working memory, its tool outputs — lives in a scope that is disjoint from Agent B's scope. The scope algebra (Chapter 1, §1.3) makes this precise:

```
scope_A = fill(instructions_A ⊕ memory_A ⊕ tools_A)
scope_B = fill(instructions_B ⊕ memory_B ⊕ tools_B)
```

Agent A generates within `scope_A`; Agent B generates within `scope_B`. Neither agent attends to the other's internal state. This is not a convention — it is enforced by the attention mask. Agent A's key-value cache entries are simply not present in Agent B's scope during generation.

Under NoPE (Chapter 1, §1.5), agent scopes compose as a free monoid. This has a direct operational consequence: agents can be added to a team, removed from a team, or reordered within a team without recomputing any other agent's KV cache. Adding a new reviewer agent to an existing planner-coder team requires only prefilling the reviewer's scope; the planner's and coder's caches remain valid. This is the multi-agent analogue of the zero-compute composition property that Chapter 5 exploits for serving.

Taint tracking (Chapter 15, §13.1) extends naturally to agent boundaries. If Agent A's scope includes spans tagged SENSITIVE, every span in Agent A's scope is tainted. Agent B's scope, being disjoint, is clean. When Agent A produces output that must be passed to Agent B, the taint label propagates to that output span. If Agent B must not receive tainted information, the `refill` primitive (Chapter 5) breaks taint: Agent A's output is regenerated under a clean scope — one that excludes the sensitive spans — before Agent B receives it. The mechanism is the same as intra-agent taint breaking; the multi-agent case adds no new machinery.

## 16.2 Inter-Agent Communication via Meanings

Agents in the generative computing stack do not communicate by passing strings. They communicate by passing **Meanings** — typed semantic objects that are independent of surface form (Chapter 8, §8.1.1).

When Agent A produces a result, that result is a Meaning with an `intent`, a `type`, and a provenance chain recording which Thoughts produced it and which scopes were active. When Agent B receives this Meaning, it enters Agent B's scope as a data-plane object — placed in `user_variables`, never in `description`. The instruction/data separation that protects against prompt injection within a single agent (Chapter 15, §13.2) applies identically at agent boundaries.

The Meaning abstraction provides a capability that string-based inter-agent protocols lack: **semantic transformation at the boundary**. Before Agent B receives Agent A's output, the orchestrating system can transform the Meaning:

```python
result_A = agent_A.execute(task)                    # Meaning with full detail
summary  = result_A.abstract()                      # same semantics, higher level
filtered = result_A.reframe("security_implications") # same facts, different lens
agent_B.receive(summary)                            # Agent B sees only what it needs
```

Each transformation produces a new Meaning. The operations — `abstract`, `concretize`, `reframe`, `paraphrase` — are the same operations defined in Chapter 8 for single-agent use. In the multi-agent setting, they serve as information-flow controls: the orchestrator decides what level of detail, what framing, and what subset of information each downstream agent receives. This is the generative computing answer to Google's A2A protocol and the Model Context Protocol (MCP) — but with structural security guarantees that text-based protocols cannot provide, because a Meaning in the data plane cannot become an instruction regardless of its content.

## 16.3 Orchestration Patterns

The dominant multi-agent patterns in the 2026 market map directly onto generative computing primitives. This section makes the mapping explicit.

### 16.3.1 Orchestrator-Worker

An orchestrator Thought decomposes a task into sub-tasks, each represented as a Meaning. Workers execute in isolated scopes. Results fan in to the orchestrator.

```
task       = Meaning("analyze codebase for security vulnerabilities")
sub_tasks  = orchestrator.decompose(task)   # [Meaning, Meaning, ...]

# Fan-out: each worker gets its own scope (Ch. 1, §1.4)
results = []
for sub_task in sub_tasks:
    scope_w = fill(worker_instructions ⊕ sub_task)
    result  = gen(scope_w)                  # isolated generation
    results.append(result)

# Fan-in: orchestrator synthesizes
scope_o = fill(orchestrator_instructions ⊕ compose(results))
final   = gen(scope_o)
```

Fan-out and fan-in are the span algebra's native operations (Chapter 1, §1.4). Each worker's scope is independent; under NoPE, workers can execute in parallel with no cache dependencies between them. The orchestrator sees worker outputs but not worker internals — the same scope isolation that protects user data within a single agent now protects inter-agent boundaries.

### 16.3.2 Evaluator-Optimizer

One agent generates; another evaluates. The evaluator's critique feeds back to the generator for iterative refinement. This is the instruct-validate-repair loop (Chapter 7, §7.3) extended across agent boundaries.

```
draft     = generator.execute(task)
critique  = evaluator.execute(draft)       # Meaning with type="evaluation"
revised   = generator.execute(task, feedback=critique)
```

The evaluator's output is a Meaning with `type` set to a Meaning representing "evaluation" — a typed annotation, not raw text. The generator receives this typed critique in its data plane and uses it as material for revision. Because the critique is a Meaning, it can be transformed before the generator receives it: `critique.concretize()` to add specificity, `critique.reframe("actionable_suggestions")` to shift from diagnosis to prescription.

The IVR loop's requirement mechanism applies: the evaluator can attach `requirements` to the critique Meaning, and the orchestrator can validate that the generator's revised output satisfies them. Multi-agent evaluation is single-agent IVR with the evaluator and generator in separate scopes.

### 16.3.3 Role-Based Teams

Agents are Thoughts with role-specific instructions and tool access. The team's behavior emerges from the composition of its members.

In the generative computing model, a role is a scope configuration: the combination of instruction spans, tool-access spans, and persona spans that define what an agent can do and how it behaves. Switching an agent's role is switching its scope — discarding the old instruction and tool spans, prefilling new ones. Under NoPE, the agent's accumulated working memory (independently scoped spans) remains valid across role switches; only the instruction context changes.

A team is a set of Thoughts with a coordination protocol. The protocol itself can be a Thought — a meta-agent whose instruction is "given these team members and their roles, decide who acts next and what they receive." The team's composition is explicit in the scope algebra: the orchestrating Thought's scope includes descriptions of each team member's capabilities (as Meanings), and the current state of the task (as a sequence of Meanings representing prior actions and their outcomes).

### 16.3.4 Hierarchical Delegation

Parent agents delegate to child agents. Each level of the hierarchy is a scope level. The parent sees child outputs but not child internals.

```
scope_parent  = fill(parent_instructions ⊕ task)
sub_task      = gen(scope_parent)            # parent decides what to delegate

scope_child   = fill(child_instructions ⊕ sub_task)
child_result  = gen(scope_child)             # child executes in isolation

scope_parent' = fill(parent_instructions ⊕ task ⊕ child_result)
final         = gen(scope_parent')           # parent integrates child's output
```

The parent's scope never includes `child_instructions` or the child's working memory — only `child_result`. This is scope isolation enforced structurally, not by convention. The hierarchy can be arbitrarily deep: a child agent can delegate to grandchild agents using the same pattern, and the grandchild's internals are invisible to both the child and the parent.

## 16.4 State and Memory in Multi-Agent Systems

Multi-agent systems require both shared and private state. The span model supports both through scope composition.

**Shared state** is a span (or set of spans) included in multiple agents' scopes. A shared knowledge base, a task description, or a conversation history can be prefilled once and composed into each agent's scope via fan-in. Under NoPE, the shared span's KV cache is computed once and reused — there is no per-agent cost for shared context.

**Private state** is a span included in exactly one agent's scope. An agent's working memory, intermediate reasoning, and tool call results live in spans that no other agent's scope includes. Privacy is structural: the spans are simply absent from other agents' attention masks.

**Persistent memory** exploits named spans. A span can be stored in the KV pool (Chapter 5) under a name and retrieved in a later session. An agent's long-term memory is a collection of named spans that are composed into its scope at the start of each session. The KV pool's eviction and lifecycle policies (Chapter 5) govern when long-term memories are discarded.

The memory lifecycle follows the span operations:

- **Creation.** `fill(new_information)` — prefill a span from new observations or tool outputs.
- **Retrieval.** `compose(memory_spans)` — include relevant memory spans in the agent's scope via fan-in.
- **Consolidation.** `gen(scope_with_memories)` — generate a summary Meaning that compresses multiple memory spans into a single span, reducing context pressure.
- **Forgetting.** Evict a memory span from the KV pool. The span's provenance record is retained (it existed, it influenced these downstream spans) but its content is no longer available for composition.

This lifecycle applies identically to single-agent and multi-agent systems. The multi-agent case adds one pattern: **memory sharing**, where an agent publishes a memory span (or a transformed version of it) to a shared namespace that other agents can retrieve. The transformation step — `memory.abstract()` or `memory.reframe(recipient_role)` — controls the granularity and framing of shared information, preventing information overload and maintaining need-to-know boundaries.

## 16.5 Security in Multi-Agent Systems

Multi-agent systems amplify every security concern from Chapter 15. A compromised or manipulated agent can inject adversarial content into another agent's context. A confused deputy problem arises when an agent with elevated tool access acts on instructions from an agent without that access. Information intended for one agent leaks to another through shared state.

The generative computing stack addresses each of these through mechanisms already described in the preceding chapters:

**Instruction/data separation at agent boundaries.** LLMON's exec mechanism (Chapter 2, §2.4) applies at every agent boundary. Agent A's output enters Agent B's scope as DATA, never as INSTRUCT. Even if Agent A has been manipulated into producing instruction-like content, that content cannot be executed by Agent B — it lives in the data plane, separated from Agent B's instructions by the same structural boundary that separates user input from system instructions.

**Taint propagation across agents.** If Agent A operates on tainted data (data derived from sensitive spans), Agent A's output carries the taint label. When Agent B receives that output, the taint label is visible to the runtime. The runtime can enforce policies: block the transfer, apply `refill` to break taint, or allow the transfer with an audit record. Taint is not an agent-level property — it is a span-level property that propagates through the scope algebra regardless of how many agents are involved.

**Scope isolation prevents privilege escalation.** Agent A's tool access is defined by the tool spans in Agent A's scope. Agent B cannot acquire Agent A's tool access by receiving Agent A's output — tool spans are not included in output Meanings. An agent's capabilities are determined by its scope configuration, set by the orchestrator, not by what other agents tell it.

**Intake pipeline at every boundary.** The intake pipeline (Chapter 10) is not a once-at-the-edge mechanism. It applies at every point where external content — including another agent's output — enters an agent's scope. Each agent boundary is a trust boundary, and the intake pipeline's parsing, validation, and Meaning extraction ensure that no raw text crosses it.

## 16.6 Open Questions

Several aspects of multi-agent orchestration in the generative computing model remain unresolved.

**Deadlock detection.** When Agent A waits for Agent B's output and Agent B waits for Agent A's output, the system deadlocks. The span algebra's dependency graph makes such cycles detectable in principle — a cycle in the scope composition graph indicates a deadlock — but the runtime does not currently implement cycle detection. Formalization of the liveness properties of multi-agent scope graphs is needed.

**Resource budgets.** A team of agents consumes tokens, latency, and KV pool capacity. How should budgets be allocated across a team? Should the orchestrator have a total budget that it distributes to workers, or should each agent have an independent budget? The KV pool's capacity management (Chapter 5) provides mechanisms for eviction under pressure, but policy — who gets evicted first when the pool is full — requires multi-agent-aware scheduling.

**Agent discovery and dynamic composition.** The patterns above assume a statically defined team. In practice, agents may need to discover other agents at runtime — finding a specialized agent for a sub-task that the orchestrator did not anticipate. This requires a registry of agent capabilities (expressed as Meanings with typed descriptions) and a matching protocol. The relationship to MCP's tool discovery and A2A's agent cards is an open design question.

**Standardized inter-agent protocols.** The generative computing stack provides Meanings as the inter-agent message format and scopes as the isolation mechanism. How these map to external standards — MCP for tool integration, A2A for cross-platform agent communication — requires a bridging layer. The bridge must preserve the structural security properties (instruction/data separation, taint tracking) when communicating with agents outside the generative computing stack that do not share these guarantees. Defining the trust model for heterogeneous multi-agent systems — some agents within the stack, some outside — is the most pressing open problem.

---

> [← Ch.15: Security Across the Stack](15-security.md) · [Table of Contents](../README.md) · [Ch.17: Frontiers →](17-frontiers.md)
