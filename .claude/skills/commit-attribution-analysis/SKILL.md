---
name: commit-attribution-analysis
description: Ranks likely introducing commits for true-positive failures using GitHub MCP commit history and diffs.
---
# Commit Attribution Analysis

## When to use
Use for true-positive failures to identify likely introducing commits from the PR range and recent target-branch history.

## Inputs
- Git context section in the system prompt: MERGE_BASE, PR_COMMITS, TARGET_RECENT_COMMITS, PR_DIFFSTAT.
- Failure evidence and suspected files from root-cause analysis.
- PR diff URL in the system prompt (derive owner/repo/PR number if needed).

## Instructions
1. Identify affected files/areas from failure evidence and root-cause analysis.
2. Use GitHub MCP tools only (do not run local git commands):
   - Call `mcp__github__get_me` to confirm access context.
   - Derive `owner`, `repo`, `pullNumber` from the PR diff URL if not provided.
   - Use `mcp__github__pull_request_read` (method: "get") to read base/head refs and head SHA.
   - Use `mcp__github__pull_request_read` (method: "get_files") to list changed files and align with `PR_DIFFSTAT`.
3. Enumerate candidate commits with `mcp__github__list_commits`:
   - List commits for the PR head ref or head SHA; stop once you reach `MERGE_BASE` to isolate PR commits.
   - List the last 20 commits on the base/target branch for recent target history.
4. Use `mcp__github__get_commit` (include_diff: true) to verify which candidate commits touch the affected files/lines.
   - If needed, use `mcp__github__get_file_contents` or `mcp__github__search_code` to capture line context for evidence.
5. Rank likely introducing commits from:
   - PR commits (highest priority)
   - Recent target-branch commits (fallback)
6. Provide evidence for each candidate (diff/file evidence). If no evidence or MCP tools are unavailable, say "Unable to attribute".

## Output format
- Likely introducing commit(s):
  - <sha> <summary> — evidence (file/line or log snippet)
- Confidence (low/medium/high)

## Mode
ultrathink
