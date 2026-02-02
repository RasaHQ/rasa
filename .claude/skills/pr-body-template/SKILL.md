---
name: pr-body-template
description: Provides a concise PR body template for CI false-positive fixes.
---
# PR Body Template

## When to use
Use when creating the PR after implementing a false-positive fix.

## Instructions
Structure the body as:
1. Summary (1-2 lines of the fix and cause)
2. Validation (commands/tests run, include `make lint` and `make types` when run)
3. Risks/Follow-ups (if any)

## Output format
- Title suggestion
- Body text with the above sections
