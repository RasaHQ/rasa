---
name: ci-evidence-pack
description: Produces a compact, job-by-job failure digest from CI failure JSON for downstream agents.
---
# CI Evidence Pack

## When to use
Use first, to turn the provided failure context JSON into actionable evidence for other agents.

## Instructions
1. Treat `failedJobLogs` as already pre-filtered (100-line context around common error terms); do not re-search full logs.
2. List each failed job with failed steps.
3. Extract the 3-10 most diagnostic log lines per job (errors, stack traces, assertions).
4. Note any timing/infra hints separately from assertion or product errors.

## Output format
- Job: <name>
- Failed steps: <comma-separated>
- Key log lines: <bullet list>
- Infra hints (if any)

## Mode
ultrathink
