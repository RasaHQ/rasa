import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

test.describe("Event details panel", () => {
  test.use({
    inspectorOptions: {
      inspectMode: true,
    },
  });

  test.describe("Flow events", () => {
    test("Clicking a flow event opens the flow event details panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Flow welcome started",
          panelTitle: "Flow event",
          slotOrFlowName: "say hello",
          containsTexts: ["Description"],
          accordionTitles: ["Event details"],
        },
      );
    });

    test("Clicking a flow completed event shows details", async ({
      inspectorPage,
    }) => {
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Flow welcome completed",
          panelTitle: "Flow event",
          slotOrFlowName: "say hello",
        },
      );
    });
  });

  test.describe("Action events", () => {
    test("Clicking an action event opens the action details panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "action_session_start",
          panelTitle: "Action event details",
          actionEventInfoVisible: true,
          accordionTitles: ["Event details"],
        },
      );
    });

    test("Clicking Waiting for user input opens action details", async ({
      inspectorPage,
    }) => {
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Waiting for user input",
          panelTitle: "Action event details",
        },
      );
    });
  });

  test.describe("Slot events", () => {
    test("Clicking a slot event opens the slot details panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.sendMessageAndAssertBotReplies(
        inspectorPage,
        "What's my balance?",
        { expectedBotMessageCount: 3 },
      );
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Slot current_balance set",
          panelTitle: "Slot event",
          slotOrFlowName: "current_balance",
          containsTexts: ["Value"],
          accordionTitles: ["Event details"],
        },
      );
    });
  });

  test.describe("Bot message events", () => {
    test("Clicking a bot response opens the agent response details panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.openBotMessageDetailsAndAssert(inspectorPage, {
        index: 0,
        accordionTitles: ["Event details"],
      });
    });

    test("Clicking a bot response after a second message shows correct panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.sendMessageAndAssertBotReplies(
        inspectorPage,
        "What's my balance?",
        { expectedBotMessageCount: 3 },
      );
      await flows.chat.openBotMessageDetailsAndAssert(inspectorPage, {
        index: 1,
        containsTexts: ["Event details"],
      });
    });
  });

  test.describe("User message events", () => {
    test("Clicking a user message opens the user message details panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.sendMessageAndAssertBotReplies(
        inspectorPage,
        "What's my balance?",
        { expectedBotMessageCount: 3 },
      );
      await flows.chat.openUserMessageDetailsAndAssert(inspectorPage, {
        index: 0,
        containsTexts: ["Predicted intents"],
        accordionTitles: ["Event details"],
      });
    });
  });

  test.describe("Panel interactions", () => {
    test("Close button dismisses event details and restores flow canvas", async ({
      inspectorPage,
    }) => {
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Flow welcome started",
          panelTitle: "Flow event",
        },
      );
      await ui.inspectorCanvas
        .assertions(inspectorPage)
        .flowCanvasIsReplacedByDetails();
      await flows.chat.closeEventDetailsAndAssertCanvasRestored(
        inspectorPage,
      );
    });

    test("Clicking the same event again deselects it", async ({
      inspectorPage,
    }) => {
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Flow welcome started",
          panelTitle: "Flow event",
        },
      );
      await flows.chat.deselectConversationEventAndAssertCanvasRestored(
        inspectorPage,
        "Flow welcome started",
      );
    });

    test("Switching between different event types updates the panel", async ({
      inspectorPage,
    }) => {
      await flows.chat.sendMessageAndAssertBotReplies(
        inspectorPage,
        "What's my balance?",
        { expectedBotMessageCount: 3 },
      );
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Flow check_balance started",
          panelTitle: "Flow event",
        },
      );
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "Slot current_balance set",
          panelTitle: "Slot event",
        },
      );
      await flows.chat.openConversationEventDetailsAndAssert(
        inspectorPage,
        {
          eventName: "action_session_start",
          panelTitle: "Action event details",
        },
      );
      await flows.chat.openBotMessageDetailsAndAssert(inspectorPage, {
        index: 1,
      });
      await flows.chat.openUserMessageDetailsAndAssert(inspectorPage, {
        index: 0,
      });
    });

    test("Event details panel is not shown when Inspect mode is off", async ({
      inspectorPage,
    }) => {
      await flows.shell.closeInspectModeAndAssertCanvasHidden(inspectorPage);
      await flows.chat.clickBotMessageAndAssertNoEventDetails(
        inspectorPage,
        0,
      );
    });
  });
});
