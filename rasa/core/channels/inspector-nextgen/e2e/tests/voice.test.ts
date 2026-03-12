import { test } from "@playwright/test";
import * as flows from "../flows/index";

test.describe("Voice functionality", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
  });

  test("Voice and send button toggle based on input text", async ({
    page,
  }) => {
    await flows.inspector.assertVoiceButtonVisibleOnLoad(page);
    await flows.inspector.typeTextAndAssertSendButtonVisible(page, "Hello");
    await flows.inspector.clearTextAndAssertVoiceButtonVisible(page);
  });

  test("Voice call workflow: start, verify active state, stop", async ({
    page,
  }) => {
    await flows.inspector.startVoiceCallAndAssertActive(page);
    await flows.inspector.assertVoiceTimerIncremented(page);
    await flows.inspector.stopVoiceCallAndAssertInactive(page);
  });
});
