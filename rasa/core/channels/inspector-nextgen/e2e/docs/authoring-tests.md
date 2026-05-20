# Authoring Inspector E2E Tests — examples

> Rules and conventions live in [../CLAUDE.md](../CLAUDE.md). This file is for worked examples only.

## Writing a simple shell test

Use the merged fixture harness and let the default `inspectorPage` fixture open the page for you.

```ts
import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";

test.describe("Inspector landing page", () => {
  test("inspect mode can be opened and closed", async ({ inspectorPage }) => {
    await flows.shell.openInspectModeAndAssertCanvas(inspectorPage);
    await flows.shell.closeInspectModeAndAssertCanvasHidden(inspectorPage);
  });
});
```

## Using `inspectorOptions`

Use `test.use({ inspectorOptions: ... })` when the setup is a page-open concern like query parameters or starting in Inspect mode.

```ts
import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";

test.describe("Event details panel", () => {
  test.use({
    inspectorOptions: {
      inspectMode: true,
    },
  });

  test("clicking a flow event opens details", async ({ inspectorPage }) => {
    await flows.chat.openConversationEventDetailsAndAssert(inspectorPage, {
      eventName: "Flow welcome started",
      panelTitle: "Flow event",
      slotOrFlowName: "say hello",
      accordionTitles: ["Event details"],
    });
  });
});
```

## Writing an inspect-mode details test

Keep the spec at the journey level. The test should not click raw locators or call low-level assertions directly.

```ts
import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";

test.use({
  inspectorOptions: {
    inspectMode: true,
  },
});

test("clicking a user message opens the user message details panel", async ({
  inspectorPage,
}) => {
  await flows.chat.sendMessageAndAssertBotReplies(
    inspectorPage,
    "What's my balance?",
  );

  await flows.chat.openUserMessageDetailsAndAssert(inspectorPage, {
    index: 0,
    containsTexts: ["Predicted intents"],
    accordionTitles: ["Event details"],
  });
});
```

## Adding a new UI triad

New UI code belongs under `actions/ui/` and should export `getLocators`, `actions`, and `assertions`.

```ts
import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => ({
  panel: page.getByTestId("example-panel"),
  openButton: page.getByRole("button", { name: "Open example" }),
});

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    open: async () => {
      await locators.openButton.click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    panelIsVisible: async () => {
      await expect(
        locators.panel,
        "Example panel should be visible",
      ).toBeVisible();
    },
  };
};
```

Then export it from `actions/ui/index.ts`:

```ts
export * as examplePanel from "./example-panel.actions";
```

## Adding a new flow

Add to the module that matches the UI area (`shell`, `chat`, `inspectPanel`, `downloads`, or `voice`). Flow files import only `@e2e/ui-actions` and `@playwright/test` — not other flow files. Specs compose namespaces when a journey crosses areas.

Flows compose UI triads and keep specs short.

```ts
import { type Page, test } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

export const openExamplePanel = async (page: Page) => {
  await test.step("Open example panel", async () => {
    await ui.examplePanel.actions(page).open();
    await ui.examplePanel.assertions(page).panelIsVisible();
  });
};
```

Then export it from `flows/index.ts`:

```ts
export * as example from "./example.flow";
```

## Reusing existing suite surfaces

Before adding a new UI helper, check whether the behavior already belongs in one of the shared surfaces:

- `inspectorShell`
- `conversationLog`
- `inspectorCanvas`
- `eventDetails`
- `viewMenu`
- `historyTimeline`
- `memoryPanel`
- `downloads`
- `voiceControls`

If the behavior already fits one of those areas, extend that triad and keep the spec itself at the flow level.

## Debugging

Scripts live in [`package.json`](../package.json). Run from `rasa/core/channels/inspector-nextgen/e2e/` with the Rasa inspector on port 5005 (or use `yarn e2e` to start it). Optional live browser tooling: [`playwright-cli`](../.claude/skills/playwright-cli/references/cli-installation.md) (global install; do not add to `package.json`).

| Symptom | First step |
| --- | --- |
| Step through / inspect live state | `PLAYWRIGHT_HTML_OPEN=never yarn e2e:only tests/<file>.test.ts --debug=cli` then `playwright-cli attach tw-…` — see [playwright-tests.md](../.claude/skills/playwright-cli/references/playwright-tests.md) |
| Playwright inspector UI | `yarn e2e:debug` or `yarn e2e:watch` |
| Assertion times out | Tighten the `expect` message; confirm finance demo state; extend triad readiness assertions |
| Flake | `yarn e2e:repeat` or `yarn e2e:repeat tests/<file>.test.ts` |
| Traces / report | `yarn report`; traces under `test-results/` per [`playwright.config.ts`](../playwright.config.ts) |
| Explore locators (no fixture) | `playwright-cli open http://localhost:5005/webhooks/inspector/inspect.html` — see [SKILL.md](../.claude/skills/playwright-cli/SKILL.md) |

Translate CLI discoveries into triads and flows; never commit raw `click eN` sequences. Full agent workflow: [`.claude/skills/playwright-cli/SKILL.md`](../.claude/skills/playwright-cli/SKILL.md).
