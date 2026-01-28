---
name: fix-test-code
description: Applies precise changes to test code to eliminate false-positive failures without weakening coverage.
---
# Fix Test Code

## When to use
Use when the failure is caused by incorrect or brittle tests.

## Instructions
1. Prefer deterministic checks over timing-based assertions.
2. Tighten fixtures to the minimal data needed.
3. Keep assertions meaningful; avoid blanket skips or xfails.

## Output format
- File(s) changed
- Test improvement summary

## Mode
ultrathink
