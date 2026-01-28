---
name: create-fix-pr
description: Creates a fix PR from the current branch targeting the failures branch.
---
# Create Fix PR

## When to use
Use after a false-positive fix is implemented on the pre-created fix branch.

## git context
- Current branch: !`git rev-parse --abbrev-ref HEAD`
- FIX_BRANCH: !`printf '%s' "${FIX_BRANCH:-<unset>}"`
- TARGET_BRANCH: !`printf '%s' "${TARGET_BRANCH:-<unset>}"`
- Changed files: !`git diff --name-only`
- Workflow-path hits: !`git diff --name-only -- '.github/workflows/' || true`

## Instructions
1. Ensure the current branch is `claude-fix-ci-on-PR-<PR-number>-comment-<comment-id>`.
2. Ensure `TARGET_BRANCH` is set to the failures branch; if missing, parse the PR number from `FIX_BRANCH` and derive it via GitHub MCP `mcp__github__pull_request_read` (method: "get") to read `headRefName` for the PR. If MCP tools are unavailable, fall back to `gh pr view "$PR_NUMBER" --json headRefName --jq '.headRefName'`.
3. If workflow-path hits are present, stop and run `restricted-change-report` (report-only; no PR).
4. Verify the diff does not touch forbidden paths: `rasa/**`, `pyproject.toml`, `poetry.lock`, `Makefile*`, `makefile_vars`, `docker/`, `scripts/`.
5. If forbidden paths are present, stop and report that the issue should be treated as true-positive.
6. Never push directly to `TARGET_BRANCH` or any non-fix branch. Only push the fix branch (e.g. `git push -u origin HEAD:refs/heads/"$FIX_BRANCH"`).
7. If a push is blocked by guardrails, do not bypass with `--force`/`--no-verify`; report the block and stop.
8. Use GitHub MCP `mcp__github__create_pull_request` with a concise title/body and `base: "$TARGET_BRANCH"` to open the PR. If MCP tools are unavailable, fall back to `gh pr create --base "$TARGET_BRANCH"`.
9. Determine reviewer:
   - Prefer `ASKER_GITHUB_LOGIN` env var if available.
   - Otherwise use PR author from `mcp__github__pull_request_read` (method: "get"). If MCP tools are unavailable, fall back to `gh pr view --json author`.
10. Request review with GitHub MCP `mcp__github__update_pull_request` and `reviewers`. If MCP tools are unavailable, use `gh pr edit --add-reviewer` (or `gh pr create --reviewer` if creating via CLI).
11. Report the PR URL.

## Output format
- PR URL
- Reviewer assigned
- Summary bullets

## Mode
ultrathink
