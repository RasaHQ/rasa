---
name: false-positive-scope-guard
description: Ensures false-positive fixes avoid forbidden areas and detect report-only paths.
---
# False-Positive Scope Guard

## When to use
Use before applying or PRing changes for a suspected false-positive.

## git diff
- Changed files: !`git diff --name-only`
- Forbidden-path hits: !`git diff --name-only -- 'rasa/' 'pyproject.toml' 'poetry.lock' 'Makefile*' 'makefile_vars' 'docker/' 'scripts/' || true`
- Workflow-path hits: !`git diff --name-only -- '.github/workflows/' || true`

## Instructions
1. Forbidden change areas: `rasa/**`, `pyproject.toml`, `poetry.lock`, `Makefile*`, `makefile_vars`, `docker/`, `scripts/`.
2. Report-only area: `.github/workflows/**` (workflow changes must be reported; no PR).
3. If the minimal fix requires changes in any forbidden area, do not proceed as false-positive.
4. Reclassify as true-positive and hand off to `ci-true-positive-reporter` with rationale.
5. If workflow-path hits are present and no forbidden paths, emit a report-only signal for downstream PR creation to stop and report.

## Output format
- Allowed change set (paths)
- Forbidden-path check: pass|fail
- Report-only: yes|no (workflow-path hits)
- If fail: reclassification note

## Mode
ultrathink
