---
name: comment-reply-format
description: Produces a clear PR comment explaining true-positive CI failures with concise structure.
---
# Comment Reply Format

## When to use
Use when replying to a PR comment explaining a true-positive failure.

## Instructions
Use this structure:
1. Cause summary (1-2 sentences).
2. Likely introducing commit(s) (bullet list with sha + evidence; say "Unable to attribute" if unknown).
3. Reproduction steps (bullet list).
4. Expected vs actual behavior (one line each).
5. Suggested next debugging focus (one line).

## Mode
ultrathink
