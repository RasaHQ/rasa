---
name: test-failure-patterns
description: Identifies common test failure patterns (assertion drift, brittle timing, fixture bugs).
---
# Test Failure Patterns

## When to use
Use when failures are localized to test code or fixtures.

## Instructions
1. Check for brittle timing, ordering, or randomness.
2. Look for incorrect expectations or stale snapshots.
3. Validate fixture/data assumptions against current behavior.

## Output format
- Pattern detected
- Example from logs or tests
- Suggested fix direction
