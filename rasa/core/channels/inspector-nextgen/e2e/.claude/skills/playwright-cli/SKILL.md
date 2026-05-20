---
name: playwright-cli
description: Use when debugging a Playwright test in inspector e2e (flaky failure, timing-out assertion, locator not found), or when exploring the inspector live to confirm locators before authoring a new test. CLI install is global only — see references/cli-installation.md. Inspector e2e only; pairs with e2e/CLAUDE.md.
allowed-tools: Bash(playwright-cli:*) Bash(npx:*) Bash(yarn:*)
---

# Playwright CLI — inspector e2e

Live browser-driving tool for **debugging** and **locator discovery**. Pairs with [`CLAUDE.md`](../../../CLAUDE.md) and [`docs/authoring-tests.md`](../../../docs/authoring-tests.md), which own authoring conventions. This skill never overrides them.

## Prerequisites

[`playwright-cli`](references/cli-installation.md) must be installed **globally** (do not add `@playwright/cli` to `package.json`).

Rasa inspector running at `BASE_URL` (default `http://localhost:5005`). Inspect page path: `/webhooks/inspector/inspect.html`. Use `yarn e2e` to start finance + tests, or `yarn e2e:only` when the server is already up.

## Hard rules — read before doing anything

1. **CLI sessions never become committed test code verbatim.** Exploration only. Translate findings into `getLocators` → `actions` / `assertions` → `flows` → thin `*.test.ts`. Pasting `playwright-cli click eN` into a spec is a convention violation.
2. **No `page.waitForTimeout`, no `page.pause()`, no `test.only` in committed code.**
3. **In `*.test.ts`, import `test`/`expect` from `@e2e/fixtures`.** In `flows/**`, import `test` from `@playwright/test` for `test.step` only.
4. **Every `expect(...)` includes a failure message** (second argument).
5. **Reuse existing UI surfaces** (`inspectorShell`, `conversationLog`, `inspectorCanvas`, `eventDetails`, `viewMenu`, `historyTimeline`, `memoryPanel`, `downloads`, `voiceControls`) before adding a new triad.

If a workflow below conflicts with these rules, the rules win.

## Decision: debug here, or open the report?

```
test failed locally        → debug here (live) or yarn report
test failed in CI only     → trace/report first; CLI only if reproducible locally
locator unknown / unstable → here (snapshot + generate-locator)
flake                      → yarn e2e:repeat; then --debug=cli if reproducible
new UI, no test yet        → here for locators, then author per CLAUDE.md
Rasa not running           → yarn e2e or start rasa inspect on finance project
```

## CLI quick reference

Inspector default URL for manual exploration:

```bash
playwright-cli open http://localhost:5005/webhooks/inspector/inspect.html
```

### Driving the page

```bash
playwright-cli goto http://localhost:5005/webhooks/inspector/inspect.html
playwright-cli snapshot
playwright-cli snapshot --depth=4
playwright-cli click e15
playwright-cli fill e5 "What's my balance?" --submit
playwright-cli close
```

### Reading the page

```bash
playwright-cli generate-locator e5 --raw
playwright-cli eval "el => el.getAttribute('data-testid')" e5
playwright-cli highlight e5
playwright-cli console
playwright-cli network
```

### Tracing a flake mid-session

```bash
playwright-cli tracing-start
# ... reproduce ...
playwright-cli tracing-stop
# open with: npx playwright show-trace <trace.zip>
```

Full command surface: `playwright-cli --help` or [playwright-tests.md](references/playwright-tests.md).

## Debugging workflows

See also [docs/authoring-tests.md#debugging](../../../docs/authoring-tests.md#debugging).

### Step through a failing test (attach)

```bash
PLAYWRIGHT_HTML_OPEN=never yarn e2e:only tests/home.test.ts --debug=cli
```

In a second terminal, `playwright-cli attach tw-…` then `snapshot` / `console` / `network`. Do not recreate fixture setup in a fresh `open` session unless the bug is URL-only.

For Playwright's built-in inspector without CLI attach: `yarn e2e:debug` or `yarn e2e:watch`.

### Locator not found / timing out

Fix usually belongs in `getLocators`, not the spec:

```bash
playwright-cli open http://localhost:5005/webhooks/inspector/inspect.html
# toggle Inspect in UI if needed
playwright-cli snapshot --depth=5
playwright-cli generate-locator e<ref> --raw
```

Prefer `getByRole` > `getByTestId` > text > CSS. Override naive `getByText` on buttons with `getByRole("button", { name: "…" })`.

### Flake

1. `yarn e2e:repeat tests/<file>.test.ts`
2. `PLAYWRIGHT_HTML_OPEN=never yarn e2e:only <file> --debug=cli`
3. Resist `waitForTimeout` — add assertions that prove readiness before the interaction (see existing flows under `flows/*.flow.ts`).

### Last failure without live CLI

```bash
yarn report
```

## CLI exploration → committed test

| Confirmed in CLI | Where it goes |
| --- | --- |
| Locator string | `getLocators` in `actions/ui/<feature>.actions.ts` |
| Click / fill / toggle | `actions(page)` in same file |
| Visible outcome | `assertions(page)` with message string |
| Multi-step journey reused across tests | `flows/<feature>.flow.ts` with `test.step` |
| Page open / inspect mode / query | `test.use({ inspectorOptions: … })` |

## Worked example — new coverage for conversation log

Goal: assert a bot message row is visible after sending (finance demo). No new triad if `conversationLog` already exposes the locator — extend it. Otherwise:

### 1. Explore (Rasa on 5005)

```bash
playwright-cli open http://localhost:5005/webhooks/inspector/inspect.html
# Inspect mode on, send "What's my balance?" in UI
playwright-cli snapshot
playwright-cli generate-locator e<bot-row-ref> --raw
```

### 2. Triad — extend `actions/ui/conversation-log.actions.ts` (or add module + `index.ts` export)

### 3. Flow — `flows/<domain>.flow.ts` if multiple tests share the journey (`shell`, `chat`, `inspectPanel`, `downloads`, `voice`)

### 4. Spec

```ts
import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";

test("bot reply appears in conversation log", async ({ inspectorPage }) => {
  await flows.chat.sendMessageAndAssertBotReplies(
    inspectorPage,
    "What's my balance?",
  );
});
```

Do **not** commit: raw CLI steps, `waitForTimeout`, `import { test } from "@playwright/test"` in specs, or `beforeEach` that duplicates `inspectorOptions`.

## References

| File | Use when |
| --- | --- |
| [references/playwright-tests.md](references/playwright-tests.md) | Yarn scripts, `--debug=cli`, reports |
| [references/cli-installation.md](references/cli-installation.md) | Global install, version constraints |

## Red flags — stop and re-read CLAUDE.md

- Pasting a CLI sequence into `*.test.ts`
- `import { test } from "@playwright/test"` in a spec
- `page.waitForTimeout` to fix a flake
- `expect(locator).toBeVisible()` without a message
- Top-level `clickFoo(page)` instead of `actions(page)`
- New triad when an existing surface already owns the behavior
