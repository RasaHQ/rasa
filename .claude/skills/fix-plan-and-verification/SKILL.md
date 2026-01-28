---
name: fix-plan-and-verification
description: Drafts a minimal false-positive fix plan plus targeted verification steps.
---
# Fix Plan and Verification

## When to use
Use before making changes for a suspected false-positive.

## Instructions
1. Identify the minimal change surface (test, fixture, config, infra knob).
2. Ensure the plan avoids forbidden paths: `rasa/**`, `pyproject.toml`, `poetry.lock`, `Makefile*`, `makefile_vars`, `docker/`, `scripts/`.
3. If the minimal fix would touch a forbidden path, stop and reclassify as true-positive.
4. If the minimal fix would touch `.github/workflows/**`, mark the plan as report-only; PR creation is blocked and a restricted-change report is required.
5. Extract the minimal pytest target(s) from failure logs (file, class, test).
6. Define the exact verification commands for those targets and commit to executing them before PR creation.
7. Always include `make lint` and `make types` in verification; passing them is mandatory before PR creation.
8. Keep scope tight; avoid full-suite runs unless necessary.

## Test target extraction
- From `FAILED tests/nlu/test_tokenizers.py::TestWhitespaceTokenizer::test_tokenize - AssertionError`, extract `tests/nlu/test_tokenizers.py::TestWhitespaceTokenizer::test_tokenize`.
- If multiple failures exist, list each target but prioritize the primary failure first.

## Verification examples
- Single test: `poetry run pytest tests/nlu/test_tokenizers.py::TestWhitespaceTokenizer::test_tokenize -x -v --tb=short`
- Test file: `poetry run pytest tests/nlu/test_tokenizers.py -x -v --tb=short`
- Name filter: `poetry run pytest tests/nlu/test_tokenizers.py -k "test_tokenize" -x -v --tb=short`

## Output format
- Planned change (1-2 bullets)
- Verification command(s) (include `make lint` and `make types`)
- Rollback/guardrails (if any)
- Forbidden-path check: pass|fail
- Report-only: yes|no (workflow changes)

## Mode
ultrathink
