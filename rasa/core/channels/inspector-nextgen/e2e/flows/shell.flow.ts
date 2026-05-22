import { type Page, test } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

const assertLandingShellState = async (page: Page) => {
  await ui.inspectorShell.assertions(page).pageIsLoaded();
  await ui.inspectorShell.assertions(page).shellIsReady();
};

export const assertLandingShellReady = async (page: Page) => {
  await test.step("Assert inspector landing shell is ready", async () => {
    await assertLandingShellState(page);
  });
};

export const openInspectModeAndAssertCanvas = async (page: Page) => {
  await test.step("Turn Inspect on", async () => {
    await ui.inspectorShell.actions(page).toggleInspectOn();
  });

  await test.step("Assert inspector canvas visible", async () => {
    await ui.inspectorCanvas.assertions(page).inspectorCanvasIsVisible();
  });
};

export const closeInspectModeAndAssertCanvasHidden = async (page: Page) => {
  await test.step("Turn Inspect off", async () => {
    await ui.inspectorShell.actions(page).toggleInspectOff();
  });

  await test.step("Assert inspector canvas hidden", async () => {
    await ui.inspectorCanvas.assertions(page).inspectorCanvasIsHidden();
  });
};

export const restartConversationAndAssertReset = async (page: Page) => {
  await test.step("Click restart conversation", async () => {
    await ui.inspectorShell.actions(page).restartConversation();
  });

  await test.step("Assert user messages cleared", async () => {
    await ui.inspectorShell.assertions(page).shellIsReady();
    await ui.conversationLog.assertions(page).userMessageCountIs(0);
  });
};
