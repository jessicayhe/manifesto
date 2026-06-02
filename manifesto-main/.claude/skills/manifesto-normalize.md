---
name: manifesto-normalize
description: Normalize all chapter metadata to the standardized frontmatter format. Splits Date into Date Created/Modified, renames Contributors to Authors, adds Version and Review History fields, and updates the README index.
---

# Manifesto Normalize

Normalize all chapter files in `chapters/` to the standardized metadata format and update the README index.

## Standardized Frontmatter Format

Every chapter MUST have this metadata block immediately after the navigation line and `---` separator, inside the chapter heading:

```
# Chapter N: Title

**Status:** Draft | In Review | Approved
**Version:** X.Y
**Authors:** Name1, Name2
**Date Created:** YYYY-MM-DD
**Date Modified:** YYYY-MM-DD
```

Followed by any optional fields (`Component`, `Scope`, `Prerequisites`) that already exist, then:

```
**Review History:**
| Reviewer | Date | Verdict | Notes |
|----------|------|---------|-------|
```

Then `---` and the chapter content.

## Steps

1. **Read all files** in `chapters/` directory matching `*.md`

2. **For each chapter file**, apply these transformations:

   a. **Rename `Contributors:` to `Authors:`** if present

   b. **Split `Date:` into two fields:**
      - `Date Created:` — use the original `Date` value
      - `Date Modified:` — use the original `Date` value (will be updated by future skill invocations)
      - Normalize all date values to `YYYY-MM-DD` format (e.g., "April 2026" becomes "2026-04-01")

   c. **Add `Version:` field if missing:**
      - If `Status` contains a version like `Draft v0.5`, extract `0.5` as the version and simplify Status to just `Draft`
      - If no version found, default to `0.1`

   d. **Add Review History table** if not already present — add it after the last metadata field, before the `---` separator

   e. **Preserve optional fields** (`Component`, `Scope`, `Prerequisites`) in their existing position

   f. **Ensure consistent spacing:** single blank line between metadata block and Review History, `---` after Review History before content

3. **Update README.md:**
   - Add `**Version:** 1.0` and `**Last Updated:** YYYY-MM-DD` to the header section if not present
   - Sync chapter titles in the Table of Contents with actual chapter titles
   - Sync description lines with the first `>` blockquote in each chapter's TOC entry

4. **Report** what was changed in each file

## Important Rules

- NEVER remove existing content — only add/transform metadata fields
- Preserve all optional fields that already exist in a chapter
- Keep navigation links (`← Previous` / `Next →`) untouched
- If a chapter already has `Date Created` and `Date Modified`, do not overwrite them
- If a chapter already has a `Version:` field, do not overwrite it
- If a chapter already has a Review History table, do not overwrite it
