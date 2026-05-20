# Inspector Nextgen E2E — Authoring Conventions

> **Audience:** humans and LLM agents writing or editing tests in `rasa/core/channels/inspector-nextgen/e2e/`.
> Read this first. For worked examples, see [docs/authoring-tests.md](docs/authoring-tests.md). For setup and run commands, see [README.md](README.md).

## Agent / CI environment

Before `yarn e2e`: from repo root, `make install` and `source .venv/bin/activate` (same as [inspector-e2e-tests action](../../../../../.github/actions/inspector-e2e-tests/action.yml)). Train finance once: `cd rasa/cli/project_templates/finance && rasa train`. Then `cd rasa/core/channels/inspector-nextgen/e2e && yarn e2e`. Server already running → `yarn e2e:only`. See [README.md](README.md#agent--ci-environment).

## The shape of a test

A test should be a thin orchestration of **flows**, not a script of raw Playwright calls.

```ts
import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";

test("inspect mode can be opened and closed", async ({ inspectorPage }) => {
  await flows.shell.openInspectModeAndAssertCanvas(inspectorPage);
  await flows.shell.closeInspectModeAndAssertCanvasHidden(inspectorPage);
});
```

- Setup goes through **fixtures** first. Prefer `test.use({ inspectorOptions: ... })` over `test.beforeEach(...)` when the setup is fixture-shaped.
- Multi-step journeys live in **flows**.
- Atomic interactions and visible outcomes live in **UI triads** under `actions/ui/`.
- Selectors live inside `getLocators(...)` in each UI module.

## Layers

| Layer | Location | Role |
| --- | --- | --- |
| `getLocators` | `actions/ui/<feature>.actions.ts` | Single locator map per feature |
| `actions` | same file | One interaction each |
| `assertions` | same file | One visible outcome each |
| `flows` | `flows/<feature>.flow.ts` | Multi-step journeys with `test.step(...)` |
| `fixtures` | `fixtures/<concern>-fixture.ts` | Shared setup and option-based lifecycle |

Each UI feature module should export **`getLocators`**, **`actions`**, and **`assertions`**.

Rules:

1. No standalone `async function clickFoo(page)` helpers outside a triad.
2. Every `expect(...)` should include a failure message.
3. Assertions do not click or type.
4. Actions may include an immediate `expect(...)` only when it directly confirms that specific interaction completed.

## Fixtures

Import `test` and `expect` from `@e2e/fixtures` in `*.test.ts` only.

This suite currently exposes `inspectorOptions`:

```ts
test.use({
  inspectorOptions: {
    inspectMode: true,
    query: "?token=None",
  },
});
```

Current option meanings:

- `open`: whether the fixture should open the inspector page. Defaults to `true`.
- `inspectMode`: whether the fixture should switch to Inspect mode after opening.
- `query`: optional query string for the inspect URL. With or without a leading `?` (e.g. `?token=None` or `token=None`).

Do not encode finance-demo conversation states into fixtures unless that becomes a deliberate suite-level pattern.

## Imports

Use aliases only in new tests and flows:

- `@e2e/ui-actions` → `actions/ui`
- `@e2e/flows` → `flows`
- `@e2e/fixtures` → merged `test` and `expect`

Import style:

- `import * as ui from "@e2e/ui-actions"`
- `import * as flows from "@e2e/flows"`

In `*.test.ts`, import `test` and `expect` from `@e2e/fixtures` only.

In `flows/**`, import `test` from `@playwright/test` for `test.step(...)`. Do not import `test` or `expect` from `@e2e/fixtures` in flow files.

## Shared UI surfaces

The suite now uses feature-scoped UI triads only. Reuse an existing surface before creating a new one:

- `inspectorShell` — navigation, view toggle, restart, send, shell readiness
- `conversationLog` — user/bot messages and conversation events (scoped under `assistant-chat`)
- `inspectorCanvas` — flow canvas, nodes, active flow name, canvas visibility
- `eventDetails`
- `viewMenu`
- `historyTimeline`
- `memoryPanel`
- `downloads`
- `voiceControls`

When a behavior already belongs to one of those surfaces, extend that module instead of creating ad-hoc helpers elsewhere.

## Flows

Flow modules under `flows/` (import via `@e2e/flows`):

| Namespace | File | Use for |
| --- | --- | --- |
| `flows.shell` | `shell.flow.ts` | Landing readiness, inspect toggle, restart + reset |
| `flows.chat` | `chat.flow.ts` | Send message, open/close event details, panel journeys |
| `flows.inspectPanel` | `inspect-panel.flow.ts` | Click collected slot (assert + click) |
| `flows.downloads` | `downloads.flow.ts` | Open popover, trigger file downloads |
| `flows.voice` | `voice.flow.ts` | Type/clear input toggles, start/stop voice call |

View switches, canvas/timeline/memory assertions, and other one-step checks use `ui.*` from specs (see `conversation-history.test.ts`, `memory.test.ts`).

**Rules:** Flow files import only `@e2e/ui-actions` and `@playwright/test`. Do **not** import other flow files — specs compose namespaces (e.g. `flows.chat.sendMessageAndAssertBotReplies` then `ui.viewMenu.actions(page).switchToMemory()`).

Flows should express product journeys, not one-line wrappers around a single click. Atomic assertions or single interactions belong in `actions/ui/*`; specs may call the triad directly when the test is a one-step smoke check (see `downloads.test.ts`).

Good flow candidates:

- open inspector landing page
- open inspect mode
- send a message and verify shell updates
- open event details for a conversation event, bot message, or user message
- close event details and restore the canvas

Flow files should import `test` from `@playwright/test` and wrap meaningful phases in `test.step(...)`.

Restart conversation clears **user** messages only; bot messages may remain in the log.

## Selectors

Selector preference order:

1. `getByRole`, `getByLabel`, and related accessible queries
2. `getByTestId`
3. text or placeholder queries when they are stable
4. CSS only as a last resort

Do not use XPath.

## Finance demo expectations

These tests run against the finance demo by default. Preserve explicit finance-demo behavior contracts that are already stable in the suite, but do not invent stronger contracts without verifying them in the current environment.

If a message-count or event-name assertion is part of the test's purpose, keep it explicit. If it is only incidental setup, prefer the narrowest assertion that proves the next step is ready.

## Do not

- `import { test } from "@playwright/test"` in `*.test.ts` (use `@e2e/fixtures` instead)
- deep imports such as `@e2e/ui-actions/*` or `@e2e/flows/*` (use the barrel aliases)
- deep relative imports when an alias exists
- `page.waitForTimeout(...)`
- `test.only` or `page.pause()` in committed code
- `if` / `try` / `catch` in test bodies unless there is no linear alternative
- bypass an existing `actions/ui/*` surface for coverage it already owns

## Where to put new code

- new UI surface: `actions/ui/<feature>.actions.ts`
- new journey: `flows/<feature>.flow.ts`
- new fixture concern: `fixtures/<concern>-fixture.ts`
- operational guidance: `README.md`
- worked examples: `docs/authoring-tests.md`

## More reading

- [docs/authoring-tests.md](docs/authoring-tests.md)
- [README.md](README.md)
- [`.claude/skills/playwright-cli/SKILL.md`](.claude/skills/playwright-cli/SKILL.md) — live debugging and locator discovery (`playwright-cli`, global install)
- [`.claude/skills/playwright-cli/references/cli-installation.md`](.claude/skills/playwright-cli/references/cli-installation.md) — global CLI install (do not add `@playwright/cli` to `package.json`)
