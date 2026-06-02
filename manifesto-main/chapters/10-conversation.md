> [← Ch.9: Decision Architecture](09-decision-architecture.md) · [Table of Contents](../README.md) · [Ch.11: Reasoning →](11-reasoning.md)

---

# Chapter 11: Conversation

**Component:** `mellea.stdlib`
**Status:** Draft
**Version:** 1.1
**Authors:** David Cox, Claude
**Date Created:** 2026-03-24
**Date Modified:** 2026-04-06

**Prerequisites:** Chapter 8 (Meaning and Thought primitives), Chapter 2 (LLMON — boundary-constrained masking for structural enforcement in Modes 1+)

**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|

---

## Goal-Oriented Conversation via the Natural Conversation Framework

This chapter applies the Meaning/Thought framework (Chapter 8) to goal-oriented conversation, governed by the Natural Conversation Framework (NCF) as a Thought grammar. The intake pipeline transforms raw user text into typed Meanings at a hard security boundary. NCF repairs are literally Meaning operations: "what do you mean?" triggers `paraphrase()`, "can you give an example?" triggers `concretize()`.

## 9.1 Application: Conversation

A conversation is a sequence of Thoughts, some verbalized, governed by a
grammar (the Natural Conversation Framework) that constrains which
Thoughts are appropriate at each point.

![Conventional multi-turn conversation vs. Meaning and Thought (NCF).
The top flow shows the standard pattern: raw text appended to chat
history, one monolithic LLM call, raw text out. The bottom flow shows
the M&T architecture: raw text is parsed into typed Meanings at the
security boundary, the NCF grammar and pragmatic state select a specific
Thought (elicit, assert, or repair), and the output feeds back as typed
state --- not appended chat
history.](media/rId35.png){width="5.833333333333333in"
height="7.034313210848644in"}

Conventional multi-turn conversation vs. Meaning and Thought (NCF). The
top flow shows the standard pattern: raw text appended to chat history,
one monolithic LLM call, raw text out. The bottom flow shows the M&T
architecture: raw text is parsed into typed Meanings at the security
boundary, the NCF grammar and pragmatic state select a specific Thought
(elicit, assert, or repair), and the output feeds back as typed state
--- not appended chat history.

### 9.3.1 The NCF as a Thought Grammar

The Natural Conversation Framework (Moore et al. 2016, 2018) defines
conversation as expandable adjacency-pair sequences. In Meaning/Thought
terms:

-   A **base pair** is two verbalized Thoughts: a first-pair-part (user
    inquiry, request) and a second-pair-part (agent answer, grant).
-   **Expansions** are additional Thoughts inserted before, within, or
    after the base pair: elicitations, repairs, closers.
-   The **sequence stack** tracks nested Thought invocations.

After any agent verbalization, six navigational Thoughts are always
available to the user:

| User action             | NCF pattern            |   Meaning operation |
| :-----------------      |  :-----------          |  :----------------  |
|  "what can you do?"     | C3: Capability Check  | (system-level) |
|  "what did you say?"    | B2.1: Repeat Request  |  `prior.repeat()` |
|  "what do you mean?"    | B2.4: Paraphrase Request |       `prior.paraphrase()` |                           
|  "okay" / "thanks"      | B4: Sequence Closer   |  (close current sequence)  |
|  "never mind"           | B5: Sequence Abort    |  (abort current sequence)  |
|  "goodbye"              | C4: Conversation Closing | (close conversation)    |                                           


The repair Thoughts are literally Meaning operations. "What do you
mean?" triggers `paraphrase()` on the prior output Meaning. "Can you
give an example?" triggers `concretize()`. The NCF content format
(response, repeat, paraphrase, example, definition) is a family of
Meaning transformations --- not special-cased handlers.

### 9.3.2 Intake Pipeline

The Intake Pipeline transforms raw user text (untrusted) into Meanings.
This is the security boundary.

    raw text ──▶ Safety Gate ──▶ Parse into Meanings ──▶ Enrich

**Safety Gate.** Deterministic filters + GuardianCheck. Hard gate --- if
it fails, the system responds with an NCF-appropriate repair Thought.

**Parse into Meanings.** Extracts: NCF action type (enum), slot values
(typed Meanings), goal signals, discourse markers. The user's raw text
is consumed by the parser and does not propagate. After this point, user
content exists only as Meaning objects in data positions (see Section
1.1 for the enforcement model across modes).

**Enrich.** Pulls from conversation state: sequence context, pragmatic
snapshot, domain state. Deterministic.

### 9.3.3 Slot Filling

A slot is a piece of information needed to fulfill a goal. Each slot has
a Meaning that describes what's needed --- not a prompt string. The
`elicit` Thought verbalizes that Meaning as a question when the system
needs to ask.

    @app.goal("find_restaurant")
    def find_restaurant():
        return goal(
            cuisine = slot(str,
                meaning=Meaning("what kind of food they want"),
                choices=lambda ctx: ctx.domain["available_cuisines"],
            ),
            distance = slot(str,
                meaning=Meaning("how far they're willing to go"),
                default="walking",
            ),
            occasion = slot(str | None,
                meaning=Meaning("whether this is a special occasion"),
                required=False,
            ),
        )

Slots get filled three ways: - **User volunteers.** Extracted during
intake parsing. - **Agent elicits.** The `elicit` Thought verbalizes the
slot's Meaning as a question. - **User elicits back.** "What are my
choices?" answered from slot metadata.

Elicitation is an NCF insert expansion --- the base pair pauses while
the elicitation pair plays out. Because the question is a verbalized
Meaning, NCF repairs work automatically: "what do you mean?" applies
`paraphrase()` to the question's Meaning.

Filling strategies (eager, batch, opportunistic, default) are
configurable per slot frame.

### 9.3.4 Pragmatic Structures

Above the NCF mechanics, pragmatic structures track what the
conversation is *about*:

**Goals.** A desired state described by Meanings. Not a procedure ---
the agent navigates to it. Goals can be declared, discovered from user
input, or spawned as sub-goals.

**Process Flows.** A sequence of steps, each expressed as a Thought. The
flow defines the happy path; the NCF handles all conversational texture
between steps.

    @app.flow("book_restaurant", depends_on="find_restaurant")
    def book_restaurant(restaurant):
        party_size = yield elicit(Meaning("how many people"))
        time = yield elicit(Meaning("what time"))
        yield confirm(Meaning("reservation details",
            variables={"restaurant": restaurant, "party_size": party_size, "time": time}))
        booking = yield action(make_reservation, restaurant, party_size, time)
        yield assert_(Meaning("reservation confirmed",
            variables={"confirmation": booking.id}))

Each `yield` is a boundary where the NCF takes over. Between yields, the
system handles repairs, clarifications, aborts. The flow author only
writes the happy path.

**Abort semantics.** When the user says "never mind" (NCF B5: Sequence
Abort) during a flow, the framework calls `.close()` on the generator,
which triggers any `finally` blocks the flow defines. Rewriting the flow
above with cleanup for a held resource:

    @app.flow("book_restaurant", depends_on="find_restaurant")
    def book_restaurant(restaurant):
        hold = yield action(hold_table, restaurant)
        try:
            party_size = yield elicit(Meaning("how many people"))
            time = yield elicit(Meaning("what time"))
            yield action(confirm_reservation, hold, party_size, time)
        finally:
            if not hold.confirmed:
                release_table(hold)

Partially-filled slot frames are preserved in the conversation state. If
the user re-initiates the same goal later, previously collected slots
can be offered as defaults.

**Hypotheses.** For reasoning-oriented conversations. Candidates are
Meanings; evidence is tracked. Thoughts like `elicit` gather evidence,
`eliminate` and `critique` (from the reasoning catalog) narrow
candidates. Not MVP.

### 9.3.5 Developer Experience

    from mellea import start_session
    from mellea.generative import Meaning
    from mellea.conversation import Conversation, goal, slot

    app = Conversation(
        description=Meaning("you help people find and book restaurants"),
    )

    @app.goal("find_restaurant")
    def find_restaurant():
        return goal(
            cuisine=slot(str, meaning=Meaning("what kind of food they want"),
                         choices=lambda ctx: ctx.domain["available_cuisines"]),
            distance=slot(str, meaning=Meaning("how far they're willing to go"),
                          default="walking"),
        )

    @find_restaurant.on_complete
    def recommend(m, result, context):
        recommendation = m.instruct(
            "Recommend a restaurant matching the user's preferences.",
            user_variables=result.slots,
            grounding_context={"restaurants": context.domain["restaurant_db"]},
            requirements=["One or two sentences."],
        )
        return Meaning(recommendation.value, type=Meaning("a restaurant recommendation"))

    m = start_session()
    app.run(m, domain={"restaurant_db": load_restaurants()})

What's absent: prompt strings for elicitation, repair handlers, turn
management. The developer declares goal state as Meanings. The framework
selects Thoughts (elicit, repair, assert, close) based on NCF grammar
and pragmatic state. Thoughts that cross into speech get verbalized ---
realized into natural language, different each time.

### 9.3.6 Safety

User text never appears in an instruction position:

|  Boundary               | What happens           | Why it's safe     |
| :-----------------      |  :-----------          |  :---------------- |
|  Safety Gate            | Raw text screened      | GuardianCheck + deterministic |
|  Parse                  | Raw text → Meanings    | Parser returns typed objects  |
|  Thought Selection      | Meanings + state → Thought  |   Grammar-driven       |
|  Thought Execution      | Meanings as `user_variables`|            Data plane  |     
|  Verbalization          | Output Meaning → text  | Only text generation point  |

Safety violations trigger NCF-appropriate Thoughts (repair, redirect,
disengage) rather than flat refusals.

---

> [← Ch.9: Decision Architecture](09-decision-architecture.md) · [Table of Contents](../README.md) · [Ch.11: Reasoning →](11-reasoning.md)
