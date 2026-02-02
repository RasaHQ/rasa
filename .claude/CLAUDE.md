## CI Failure Auto-Fix Orchestrator

When invoked by the `analyse-and-fix-ci-test-failures` workflow, follow this modular sequence and delegate to sub-agents as needed:

1. Start with `ci-failure-triager`: build a concise evidence pack from the provided JSON, hypothesize root cause, and classify. If true-positive, include likely introducing commits via `commit-attribution-analysis`.
2. If classification is **false-positive**:
   - Run `ci-false-positive-fixer` to draft a minimal fix plan, apply changes, and prepare targeted verification.
   - Run `run-local-verification` to execute the targeted tests. If verification fails, revise the fix and re-verify (max 3 attempts); if still failing, reclassify as true-positive and stop.
   - If the diff includes `.github/workflows/**`, run `restricted-change-report` (summary + patch diff) and stop; do not open a PR.
   - Otherwise, run `ci-fix-pr-creator` to open a PR from `claude-fix-ci-on-PR-<PR-number>-comment-<comment-id>` targeting the failures branch (`TARGET_BRANCH`) and assign only the asker as reviewer.
   - Guardrail: false-positive fixes must not touch `rasa/**`, `pyproject.toml`, `poetry.lock`, `Makefile*`, `makefile_vars`, `docker/`, `scripts/`. If required, reclassify as true-positive.
   - Guardrail: `.github/workflows/**` changes are report-only and must not be PR'd.
   - Guardrail: never push directly to `TARGET_BRANCH` or any non-fix branch. Only push the fix branch and open a PR.
3. If classification is **true-positive**:
   - Run `ci-true-positive-reporter` to draft a PR comment that explains the bug, includes likely introducing commits, and provides reproduction steps.
4. Keep outputs concise and evidence-based; avoid speculative changes.

Notes:
- CI failure context is available in ${WORKSPACE}/failure-data.json file.
- Agents and skills are discovered automatically from `.claude/agents` and `.claude/skills`.
