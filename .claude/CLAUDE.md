## CI Failure Auto-Fix

When invoked by `analyse-and-fix-ci-test-failures`, follow this phased sequence. Keep all outputs to short structured bullets. Never repeat evidence already in context.

### Phase 1 -- Triage (always run first)

1. Parse the CI failure context from the prompt. Treat logs as already pre-filtered
   (5-line context around errors); do not re-search full logs.
2. List each failed job with failed steps.
3. Extract the 3-10 most diagnostic log lines per job (errors, stack traces, assertions).
4. Note any timing/infra hints (timeouts, ECONNRESET, OOM) separately.
5. Map errors to code locations; distinguish test failures vs infra issues vs product defects.
6. Classify: **false-positive** | **true-positive** | **both**.
   - false-positive: flaky infra, test bugs, bad fixtures/config.
   - true-positive: product code behavior violates expectations. Be conservative;
     maximise signal, minimise noise.
7. If true-positive: attribute likely introducing commits using the git context from the
   prompt (MERGE_BASE, PR_COMMITS, TARGET_RECENT_COMMITS, PR_DIFFSTAT).
   Use focused commands only: `git log --oneline -- <path>` and
   `git show --stat <sha>` for 1-2 key files.
8. If a plausible fix requires forbidden paths (see Guardrails), classify as true-positive.

**Output**: summary (1-2 sentences), root cause (bullets), classification + confidence,
evidence, likely introducing commit(s) if true-positive, recommended next phase.

### Phase 2 -- False-positive fix + PR (only if classification includes false-positive)

1. Identify the minimal change surface (test, fixture, config, infra knob).
2. Extract the narrowest pytest target(s) from failure logs.
3. Implement the fix:
   - Infra noise (timeouts, ECONNRESET, OOM): stabilize with retries/timeouts/caching.
   - Test code: prefer deterministic checks over timing-based assertions; keep assertions
     meaningful; avoid blanket skips or xfails.
   - Test data/config: update fixtures/configs to match current expected behavior; keep
     changes minimal.
4. **Scope check** (run after making changes):
   - `git diff --name-only` -- if any forbidden paths appear, reclassify as true-positive
     and stop.
   - `git diff --name-only -- '.github/workflows/'` -- if workflow-path hits and no
     forbidden paths, output a report with `git diff -U3` and stop (no PR).
5. Verify: `poetry run pytest <target> -x -v --tb=short 2>&1 | tail -50`.
   If fail, revise (max 3 attempts). If still failing, reclassify as true-positive and stop.
   If a test exceeds 5 min or total verification exceeds 10 min, mark inconclusive and stop.
6. **Scope re-check**: repeat step 4 after verification passes.
7. Ensure current branch matches `FIX_BRANCH` env var. If `TARGET_BRANCH` is unset,
   derive it: `gh pr view "$PR_NUMBER" --json headRefName --jq '.headRefName'`.
8. Push fix branch only: `git push -u origin HEAD:refs/heads/"$FIX_BRANCH"`.
   Never push to TARGET_BRANCH. If blocked by guardrails, report and stop.
9. Create PR: `gh pr create --base "$TARGET_BRANCH"` with concise title and body
   (summary of fix, validation commands run, risks/follow-ups).
10. Assign reviewer: `$ASKER_GITHUB_LOGIN` env var, or fall back to
    `gh pr view --json author`.
11. Output: PR URL, reviewer, summary.

### Phase 3 -- True-positive report (only if classification includes true-positive)

1. Generate minimal reproduction steps (smallest test or CLI command; be explicit about
   inputs and commands).
2. Post a PR comment via `gh pr comment` with this structure:
   - Cause summary (1-2 sentences).
   - Likely introducing commit(s) (bullet list with sha + evidence;
     say "Unable to attribute" if unknown).
   - Reproduction steps (bullet list).
   - Expected vs actual behavior (one line each).
   - Suggested next debugging focus (one line).

### Phase 4 -- Both (only if classification = both)

Execute Phase 2 for the false-positive failure(s), then Phase 3 for the true-positive
failure(s).

### Guardrails (always in effect)

- **Forbidden change areas**: `rasa/**`, `pyproject.toml`, `poetry.lock`, `Makefile*`,
  `makefile_vars`, `docker/`, `scripts/`. If a fix requires these, reclassify as
  true-positive.
- **Workflow report-only**: `.github/workflows/**` changes must be reported (summary +
  patch diff) but never PR'd.
- **Branch safety**: never push to `TARGET_BRANCH` or any non-fix branch. Only push
  `FIX_BRANCH`. Never use `--force` or `--no-verify`.
