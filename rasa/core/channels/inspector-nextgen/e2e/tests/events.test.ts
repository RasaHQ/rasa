import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

const initialConversationEvents = [
  "action_session_start",
  "Waiting for user input",
  "System flow pattern_session_start started",
  "System flow pattern_session_start completed",
  "Flow welcome started",
  "Flow welcome completed",
];

const postBalanceConversationEvents = [
  "Flow check_balance started",
  "Slot current_balance set",
  "utter_current_balance",
  "Flow check_balance completed",
  "System flow pattern_completed started",
  "Slot continue_conversation is cleared",
  "utter_ask_continue_conversation",
];

const greetingFlowNodes = ["Start", "utter_greeting"];
const patternCompletedFlowNodes = ["Start", "utter_can_do_something_else"];

test.describe("Inspector events", () => {
  test.use({
    inspectorOptions: {
      inspectMode: true,
    },
  });

  test("Flow events are visible in inspect mode", async ({ inspectorPage }) => {
    await ui.inspectorCanvas
      .assertions(inspectorPage)
      .canvasWithNodesIsVisible();

    await test.step("Assert the initial finance demo flow state", async () => {
      await ui.inspectorCanvas
        .assertions(inspectorPage)
        .activeFlowNameIsVisible("say hello");

      for (const nodeName of greetingFlowNodes) {
        await ui.inspectorCanvas
          .assertions(inspectorPage)
          .flowNodeIsVisible(nodeName);
      }

      for (const eventName of initialConversationEvents) {
        await ui.conversationLog
          .assertions(inspectorPage)
          .conversationEventIsVisible(eventName);
      }
    });

    await flows.chat.sendMessageAndAssertBotReplies(
      inspectorPage,
      "What's my balance?",
      { expectedBotMessageCount: 3 },
    );

    await test.step("Assert balance check and pattern completed flow state", async () => {
      for (const eventName of postBalanceConversationEvents) {
        await ui.conversationLog
          .assertions(inspectorPage)
          .conversationEventIsVisible(eventName);
      }

      await ui.inspectorCanvas
        .assertions(inspectorPage)
        .activeFlowNameIsVisible("pattern completed");

      for (const nodeName of patternCompletedFlowNodes) {
        await ui.inspectorCanvas
          .assertions(inspectorPage)
          .flowNodeIsVisible(nodeName);
      }
    });
  });
});
