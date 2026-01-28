---
name: run-local-verification
description: Execute targeted tests locally to verify a false-positive fix resolves the failure.
---
# Run Local Verification

## When to use
Use after applying a false-positive fix and before PR creation.

## Inputs
- Failed test identifiers or logs from triage
- Fix summary or changed files from fixer

## Instructions
1. Extract the narrowest pytest target(s) from failure evidence (file, class, test).
2. Run targeted verification with `poetry run pytest <target> -x -v --tb=short`.
3. Capture command, exit code, runtime, and a short output snippet (use `2>&1 | tail -50`, ensure exit code is preserved).
4. If the run fails, summarize the failure and suggest the next adjustment; do not proceed to PR creation.
5. If a test exceeds 5 minutes or total verification exceeds 10 minutes, mark as inconclusive and stop.

## Output format
- Target(s) tested
- Command(s) run
- Result: pass|fail|inconclusive
- Evidence snippet
- Follow-up recommendation

## Mode
ultrathink
