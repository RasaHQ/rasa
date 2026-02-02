---
name: fix-test-data-config
description: Adjusts test data, configs, or setup to remove noise while preserving intended test coverage.
---
# Fix Test Data or Config

## When to use
Use when failures are due to outdated fixtures, mismatched configs, or setup drift.

## Instructions
1. Update fixtures/configs to match current expected behavior.
2. Keep changes minimal and well-scoped.
3. Avoid altering production defaults unless required.

## Output format
- File(s) changed
- Reason for change
