---
name: restricted-change-report
description: Reports workflow-file changes with summary and patch diff.
---
# Restricted Change Report

## When to use
Use when changes touch `.github/workflows/**` and PR creation is blocked.

## git diff
- Changed files: !`git diff --name-only`
- Workflow-path hits: !`git diff --name-only -- '.github/workflows/' || true`
- Patch diff: !`git diff`

## Instructions
1. Confirm workflow-path hits are present.
2. Summarize why the workflow change is needed (1-2 bullets).
3. Output a patch-style diff for the current changes; do not push or open a PR.
4. If forbidden paths are also present, state that the change is out of scope and should be treated as true-positive.

## Output format
- Report-only: yes (workflow changes)
- Changed files
- Rationale (1-2 bullets)
- Patch diff

## Mode
ultrathink
