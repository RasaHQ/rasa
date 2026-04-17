import { test } from "@playwright/test";
import * as flows from "../flows/index";
import * as actions from "../actions/index";

test.describe("Event details panel", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
    await flows.inspector.toggleInspectOnAndAssertFlowPanel(page);
  });

  test.describe("Flow events", () => {
    test("Clicking a flow event opens the flow event details panel", async ({
      page,
    }) => {
      await flows.inspector.clickConversationEventAndAssertPanel(
        page,
        "Flow welcome started",
        "Flow event",
      );

      await test.step("Assert flow name is displayed", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventSlotOrFlowName("say hello");
      });

      await test.step("Assert description field is present", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelContainsText("Description");
      });

      await test.step("Assert Event details accordion exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsAccordion("Event details");
      });
    });

    test("Clicking a flow completed event shows details", async ({ page }) => {
      await flows.inspector.clickConversationEventAndAssertPanel(
        page,
        "Flow welcome completed",
        "Flow event",
      );

      await test.step("Assert flow name is displayed", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventSlotOrFlowName("say hello");
      });
    });
  });

  test.describe("Action events", () => {
    test("Clicking an action event opens the action details panel", async ({
      page,
    }) => {
      await flows.inspector.clickConversationEventAndAssertPanel(
        page,
        "action_session_start",
        "Action event details",
      );

      await test.step("Assert action event info section is displayed", async () => {
        await actions.inspector.assertions(page).assertActionEventInfo();
      });

      await test.step("Assert Event details accordion exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsAccordion("Event details");
      });
    });

    test("Clicking Waiting for user input opens action details", async ({
      page,
    }) => {
      await flows.inspector.clickConversationEventAndAssertPanel(
        page,
        "Waiting for user input",
        "Action event details",
      );
    });
  });

  test.describe("Slot events", () => {
    test("Clicking a slot event opens the slot details panel", async ({
      page,
    }) => {
      await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
      await actions.inspector.assertions(page).assertBotMessageCount(3);

      await test.step("Wait for slot event", async () => {
        await flows.inspector.assertConversationEventVisible(
          page,
          "Slot current_balance set",
        );
      });

      await flows.inspector.clickConversationEventAndAssertPanel(
        page,
        "Slot current_balance set",
        "Slot event",
      );

      await test.step("Assert slot name is displayed", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventSlotOrFlowName("current_balance");
      });

      await test.step("Assert slot value section exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelContainsText("Value");
      });

      await test.step("Assert Event details accordion exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsAccordion("Event details");
      });
    });
  });

  test.describe("Bot message events", () => {
    test("Clicking a bot response opens the agent response details panel", async ({
      page,
    }) => {
      await flows.inspector.clickBotMessageAndAssertPanel(page, 0);

      await test.step("Assert Event details accordion exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsAccordion("Event details");
      });
    });

    test("Clicking a bot response after a second message shows correct panel", async ({
      page,
    }) => {
      await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
      await actions.inspector.assertions(page).assertBotMessageCount(3);

      await flows.inspector.clickBotMessageAndAssertPanel(page, 1);

      await test.step("Assert agent response details panel content", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelContainsText("Event details");
      });
    });
  });

  test.describe("User message events", () => {
    test("Clicking a user message opens the user message details panel", async ({
      page,
    }) => {
      await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
      await actions.inspector.assertions(page).assertBotMessageCount(3);

      await flows.inspector.clickUserMessageAndAssertPanel(page, 0);

      await test.step("Assert Predicted intents section exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelContainsText("Predicted intents");
      });

      await test.step("Assert Event details accordion exists", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsAccordion("Event details");
      });
    });
  });

  test.describe("Panel interactions", () => {
    test("Close button dismisses event details and restores flow canvas", async ({
      page,
    }) => {
      await flows.inspector.clickConversationEventAndAssertPanel(
        page,
        "Flow welcome started",
        "Flow event",
      );

      await flows.inspector.assertFlowCanvasReplacedByEventDetails(page);
      await flows.inspector.closeEventDetailsAndAssertHidden(page);
      await flows.inspector.assertFlowCanvasRestored(page);
    });

    test("Clicking the same event again deselects it", async ({ page }) => {
      await test.step("Select the event", async () => {
        await actions.inspector
          .actions(page)
          .clickConversationEvent("Flow welcome started");
      });

      await test.step("Assert event details visible", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelVisible("Flow event");
      });

      await test.step("Click the same event to deselect", async () => {
        await actions.inspector
          .actions(page)
          .clickConversationEvent("Flow welcome started");
      });

      await test.step("Assert event details hidden and canvas restored", async () => {
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelHidden();
        await flows.inspector.assertFlowCanvasRestored(page);
      });
    });

    test("Switching between different event types updates the panel", async ({
      page,
    }) => {
      await flows.inspector.sendMessageAndAssert(page, "What's my balance?");
      await actions.inspector.assertions(page).assertBotMessageCount(3);

      await test.step("Open flow event details", async () => {
        await flows.inspector.clickConversationEventAndAssertPanel(
          page,
          "Flow check_balance started",
          "Flow event",
        );
      });

      await test.step("Switch to slot event", async () => {
        await actions.inspector
          .actions(page)
          .clickConversationEvent("Slot current_balance set");
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelVisible("Slot event");
      });

      await test.step("Switch to action event", async () => {
        await actions.inspector
          .actions(page)
          .clickConversationEvent("action_session_start");
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelVisible("Action event details");
      });

      await test.step("Switch to bot message", async () => {
        await actions.inspector.actions(page).clickBotMessage(1);
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelVisible("Agent response details");
      });

      await test.step("Switch to user message", async () => {
        await actions.inspector.actions(page).clickUserMessage(0);
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelVisible("User message details");
      });
    });

    test("Event details panel is not shown when Inspect mode is off", async ({
      page,
    }) => {
      await test.step("Turn off inspect mode", async () => {
        await actions.inspector.actions(page).toggleInspectOff();
      });

      await test.step("Assert inspector canvas is hidden", async () => {
        await actions.inspector
          .assertions(page)
          .assertInspectorCanvasHidden();
      });

      await test.step("Click bot message - should not open details panel", async () => {
        await actions.inspector.actions(page).clickBotMessage(0);
        await actions.inspector
          .assertions(page)
          .assertEventDetailsPanelHidden();
      });
    });
  });
});
