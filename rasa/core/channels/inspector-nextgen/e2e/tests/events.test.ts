import { test } from "@playwright/test";
import * as flows from "../flows/index";
import * as actions from "../actions/index";

test.describe("Inspector Landing page", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
  });

  test("Flow events are visible in the inspector mode", async ({ page }) => {
    await flows.inspector.toggleInspectOnAndAssertFlowPanel(page);

    await test.step("Assert active flow node", async () => {
      await flows.inspector.assertActiveFlowName(page, "say hello");
      await flows.inspector.assertFlowNodeVisible(page, "Start");
      await flows.inspector.assertFlowNodeVisible(page, "utter_greeting");
    });

    await test.step("Assert conversation event", async () => {
      await flows.inspector.assertConversationEventVisible(
        page,
        "action_session_start",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "Waiting for user input",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "System flow pattern_session_start started",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "System flow pattern_session_start completed",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "Flow welcome started",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "Flow welcome completed",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "Waiting for user input",
      );
    });

    await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
    await actions.inspector.assertions(page).assertBotMessageCount(3);

    await test.step("Assert flow nodes are visible", async () => {
      await flows.inspector.assertActiveFlowName(page, "pattern completed");
      await flows.inspector.assertFlowNodeVisible(page, "Start");
      await flows.inspector.assertFlowNodeVisible(page, "if...");
      await flows.inspector.assertFlowNodeVisible(page, "utter_closing_words");
    });
    await test.step("Assert conversation events are visible", async () => {
      await flows.inspector.assertConversationEventVisible(
        page,
        "Flow check_balance started",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "Slot current_balance set",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "utter_current_balance",
      );
      await flows.inspector.assertConversationEventVisible(
        page,
        "Flow check_balance completed",
      );
    });
  });
});
