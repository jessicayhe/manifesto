---
name: manifesto-review
description: Transition a chapter's review status (Draft/In Review/Approved) and log the review in the chapter's Review History table.
---

# Manifesto Review

Transition a chapter's review status and log the reviewer in the Review History table.

## Arguments

This skill expects arguments in the format:
```
/manifesto-review <chapter-number> "<reviewer-name>" <verdict>
```

Where `<verdict>` is one of: `approve`, `request-changes`, `in-review`

Examples:
- `/manifesto-review 01 "Jane Smith" approve`
- `/manifesto-review 05 "Bob Lee" request-changes`
- `/manifesto-review 08 "Alice Chen" in-review`

If arguments are not provided, ask the user interactively:
1. Which chapter? (show numbered list)
2. Reviewer name?
3. Verdict? (approve / request-changes / in-review)
4. Any notes? (optional)

## Status Workflow

Valid transitions:
- `Draft` → `In Review` (verdict: `in-review`)
- `In Review` → `Approved` (verdict: `approve`)
- `In Review` → `Draft` (verdict: `request-changes` — sends back for rework)
- `Approved` → `In Review` (verdict: `in-review` — reopened for changes)

If the requested transition is invalid, explain the current status and valid options.

## Steps

1. **Locate and read the chapter file**

2. **Validate the status transition** — check current `Status:` against the requested verdict

3. **Update `Status:` field:**
   - `in-review` → set to `In Review`
   - `approve` → set to `Approved`
   - `request-changes` → set to `Draft`

4. **Append a row to the Review History table:**
   ```
   | Reviewer Name | YYYY-MM-DD | Verdict | Notes |
   ```
   Where Verdict is one of: `Approved`, `Changes Requested`, `Review Started`

5. **Update `Date Modified:`** to today's date

6. **Bump version:**
   - If verdict is `approve`: major version bump (`0.6` → `1.0`, `1.4` → `2.0`)
   - Otherwise: minor version bump (`0.5` → `0.6`)

7. **Report** the status change and updated metadata

## Important Rules

- Do NOT modify any content in the chapter body
- Do NOT skip the Review History log — every review action must be recorded
- If the chapter lacks a Review History table, add one before appending the row
- If the chapter does not have standardized metadata, suggest running `/manifesto-normalize` first
