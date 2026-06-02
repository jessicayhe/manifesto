---
name: manifesto-contribute
description: Add a new chapter to the manifesto or update an existing one. Creates files with standardized frontmatter, updates navigation links, and maintains the README index.
---

# Manifesto Contribute

Guide the addition of new chapters or significant updates to existing chapters.

## Arguments

```
/manifesto-contribute new "<chapter-title>"
/manifesto-contribute update <chapter-number>
```

Examples:
- `/manifesto-contribute new "Distributed Span Scheduling"`
- `/manifesto-contribute update 05`

If no arguments, ask the user:
1. New chapter or update existing?
2. If new: chapter title and which Part (I, II, or III) it belongs to
3. If update: which chapter number

## New Chapter Mode

### Steps

1. **Determine the next chapter number** — scan `chapters/` for the highest-numbered file, increment by 1, zero-pad to 2 digits

2. **Determine placement** — ask the user which Part the chapter belongs to:
   - Part I: The Generative Computing Compute Model (chapters 1-6)
   - Part II: High-Level Generative Computing Patterns (chapters 7-13)
   - Part III: Generative Computing Systems (chapters 14+)

3. **Ask for chapter details:**
   - One-line description (for the README TOC `>` blockquote)
   - Author name(s)
   - Prerequisites (optional)
   - Component/Scope (optional)

4. **Create the chapter file** at `chapters/XX-slug.md` with:

   ```markdown
   > [← Ch.N-1: Previous Title](NN-previous.md) · [Table of Contents](../README.md) · [Ch.N+1: Next Title →](NN-next.md)

   ---

   # Chapter N: Title

   **Status:** Draft
   **Version:** 0.1
   **Authors:** Author Names
   **Date Created:** YYYY-MM-DD
   **Date Modified:** YYYY-MM-DD

   **Review History:**
   | Reviewer | Date | Verdict | Notes |
   |----------|------|---------|-------|

   ---

   *Chapter content goes here.*
   ```

   If the chapter is the last one, the "Next" navigation link should be omitted or set to a placeholder.

5. **Update navigation links** in the previous last chapter to point to the new chapter

6. **Update README.md:**
   - Add the new chapter entry in the correct Part section
   - Bump manifesto minor version
   - Update `Last Updated` date

7. **Report** what was created and updated

## Update Existing Chapter Mode

After the user has made content edits to a chapter:

### Steps

1. **Read the chapter file** to check current metadata

2. **Update `Date Modified:`** to today's date

3. **Bump chapter minor version** (e.g., `0.5` → `0.6`)

4. **Sync README index** — if the chapter title or description blockquote changed, update the README TOC entry to match

5. **Update manifesto `Last Updated`** date in README

6. **Report** what was updated

## Important Rules

- For new chapters, ALWAYS use the standardized frontmatter format
- NEVER renumber existing chapters — new chapters get the next available number
- Keep the README TOC description consistent with the chapter's opening blockquote
- The slug in the filename should be a kebab-case version of the chapter title (e.g., `16-distributed-span-scheduling.md`)
