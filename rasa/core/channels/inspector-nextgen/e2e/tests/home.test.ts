import { test } from "@playwright/test";
import * as flows from "../flows/index";
import * as actions from "../actions/index";

test.describe("Inspector Landing page", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
  });

  test("Inspect toggle on shows flow panel, off hides canvas", async ({
    page,
  }) => {
    await flows.inspector.toggleInspectOnAndAssertFlowPanel(page);
    await flows.inspector.toggleInspectOffAndAssertCanvasHidden(page);
  });

  test("Restart conversation resets conversation area", async ({ page }) => {
    const message = "Hi there!";
    await flows.inspector.sendMessageAndAssert(page, message);
    await flows.inspector.restartConversationAndAssert(page);
    await actions.inspector.assertions(page).assertUserMessageCount(0);
  });

  test("Send message with Enter key adds user message and clears input", async ({
    page,
  }) => {
    await flows.inspector.sendMessageWithEnterAndAssert(page, "Hello");
    await actions.inspector.assertions(page).assertUserMessageCount(1);
  });

  test("URL with token=None loads without error", async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page, {
      query: "?token=None",
    });
  });
});
