---
name: manifesto-add-author
description: Add a contributor to a chapter's Authors field, update Date Modified, and bump the chapter version.
---

# Manifesto Add Author

Add a person to a chapter's `Authors:` field.

## Arguments

This skill expects arguments in the format:
```
/manifesto-add-author <chapter-number> "<author-name>"
```

Examples:
- `/manifesto-add-author 06 "Nima Dehmamy"`
- `/manifesto-add-author 01 "Jane Smith"`

If arguments are not provided, ask the user interactively:
1. Which chapter? (show numbered list of chapters)
2. What is the author's name?

## Steps

1. **Locate the chapter file** — find `chapters/XX-*.md` matching the chapter number (zero-padded to 2 digits)

2. **Read the chapter file** and find the `**Authors:**` line

3. **Check if author already present** — if the name already appears in the Authors field, inform the user and stop (no changes needed)

4. **Add the author name** — append to the existing Authors list, separated by comma and space. Example:
   - Before: `**Authors:** David Cox, Claude`
   - After: `**Authors:** David Cox, Claude, Jane Smith`

5. **Update `Date Modified:`** to today's date (`YYYY-MM-DD`)

6. **Bump the chapter version** — increment the minor version:
   - `0.5` → `0.6`
   - `1.3` → `1.4`

7. **Report** what was changed

## Important Rules

- Do NOT modify any content in the chapter body
- Do NOT modify `Date Created:`
- If the chapter does not have standardized metadata (no `Date Modified:` or `Version:` fields), suggest running `/manifesto-normalize` first
