import { test } from "@playwright/test";
import * as flows from "../flows/index";
import * as actions from "../actions/index";

test.describe("Conversation History view", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
    await flows.inspector.toggleInspectOnAndAssertFlowPanel(page);
  });

  test("Switching to History view shows the flow timeline", async ({
    page,
  }) => {
    await flows.inspector.switchToHistoryView(page);
    await flows.inspector.assertFlowTimelineVisible(page);
  });

  test("Flow timeline shows completed flows from session start", async ({
    page,
  }) => {
    await flows.inspector.switchToHistoryView(page);
    await flows.inspector.assertFlowTimelineVisible(page);

    await test.step("Assert welcome flow is in the timeline", async () => {
      await flows.inspector.assertFlowTimelineItemVisible(page, "say hello");
      await flows.inspector.assertFlowTimelineItemHasStatus(
        page,
        "say hello",
        "Completed",
      );
    });
  });

  test("Flow timeline updates after user triggers a new flow", async ({
    page,
  }) => {
    await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
    await actions.inspector.assertions(page).assertBotMessageCount(2);

    await flows.inspector.switchToHistoryView(page);
    await flows.inspector.assertFlowTimelineVisible(page);

    await test.step("Assert check_balance flow appears in timeline", async () => {
      await flows.inspector.assertFlowTimelineItemVisible(
        page,
        "check account balance",
      );
      await flows.inspector.assertFlowTimelineItemHasStatus(
        page,
        "check account balance",
        "Completed",
      );
    });

    await test.step("Assert welcome flow is still in the timeline", async () => {
      await flows.inspector.assertFlowTimelineItemVisible(page, "say hello");
    });
  });

  test("Switching between views preserves state", async ({ page }) => {
    await flows.inspector.switchToHistoryView(page);
    await flows.inspector.assertFlowTimelineVisible(page);

    await test.step("Switch to Active Flow view", async () => {
      await flows.inspector.switchToActiveFlowView(page);
      await actions.inspector.assertions(page).assertFlowCanvasVisible();
    });

    await test.step("Switch back to History - timeline still visible", async () => {
      await flows.inspector.switchToHistoryView(page);
      await flows.inspector.assertFlowTimelineVisible(page);
    });
  });

  test("Restarting conversation resets the timeline and shows the new session flows", async ({
    page,
  }) => {
    await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
    await actions.inspector.assertions(page).assertBotMessageCount(2);

    await flows.inspector.switchToHistoryView(page);
    await flows.inspector.assertFlowTimelineItemVisible(
      page,
      "check account balance",
    );

    await test.step("Restart conversation", async () => {
      await flows.inspector.restartConversationAndAssert(page);
    });

    await test.step("Assert timeline no longer contains pre-restart flows", async () => {
      await flows.inspector.switchToHistoryView(page);
      await flows.inspector.assertFlowTimelineVisible(page);
      await flows.inspector.assertFlowTimelineItemVisible(page, "say hello");
      await flows.inspector.assertFlowTimelineItemNotVisible(
        page,
        "check account balance",
      );
    });
  });
});
