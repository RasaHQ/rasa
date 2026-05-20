import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

test.describe("Conversation History view", () => {
  test.use({
    inspectorOptions: {
      inspectMode: true,
    },
  });

  test("Switching to History view shows the flow timeline", async ({
    inspectorPage,
  }) => {
    await ui.viewMenu.actions(inspectorPage).switchToHistory();
    await ui.historyTimeline.assertions(inspectorPage).timelineIsVisible();
  });

  test("Flow timeline shows completed flows from session start", async ({
    inspectorPage,
  }) => {
    await ui.viewMenu.actions(inspectorPage).switchToHistory();
    await ui.historyTimeline.assertions(inspectorPage).timelineIsVisible();

    await test.step("Assert welcome flow is in the timeline", async () => {
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemIsVisible("say hello");
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemHasStatus("say hello", "Completed");
    });
  });

  test("Flow timeline updates after user triggers a new flow", async ({
    inspectorPage,
  }) => {
    await flows.chat.sendMessageAndAssertBotReplies(
      inspectorPage,
      "What's my balance?",
      { expectedBotMessageCount: 3 },
    );

    await ui.viewMenu.actions(inspectorPage).switchToHistory();
    await ui.historyTimeline.assertions(inspectorPage).timelineIsVisible();

    await test.step("Assert check account balance appears in the timeline", async () => {
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemIsVisible("check account balance");
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemHasStatus("check account balance", "Completed");
    });

    await test.step("Assert welcome flow is still in the timeline", async () => {
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemIsVisible("say hello");
    });
  });

  test("Switching between views preserves state", async ({ inspectorPage }) => {
    await ui.viewMenu.actions(inspectorPage).switchToHistory();
    await ui.historyTimeline.assertions(inspectorPage).timelineIsVisible();

    await test.step("Switch to Active Flow view", async () => {
      await ui.viewMenu.actions(inspectorPage).switchToActiveFlow();
      await ui.inspectorCanvas.assertions(inspectorPage).flowCanvasIsVisible();
    });

    await test.step("Switch back to History - timeline still visible", async () => {
      await ui.viewMenu.actions(inspectorPage).switchToHistory();
      await ui.historyTimeline.assertions(inspectorPage).timelineIsVisible();
    });
  });

  test("Restarting conversation resets the timeline and shows the new session flows", async ({
    inspectorPage,
  }) => {
    await flows.chat.sendMessageAndAssertBotReplies(
      inspectorPage,
      "What's my balance?",
      { expectedBotMessageCount: 3 },
    );

    await ui.viewMenu.actions(inspectorPage).switchToHistory();
    await ui.historyTimeline
      .assertions(inspectorPage)
      .itemIsVisible("check account balance");

    await test.step("Restart conversation", async () => {
      await flows.shell.restartConversationAndAssertReset(inspectorPage);
    });

    await test.step("Assert timeline no longer contains pre-restart flows", async () => {
      await ui.viewMenu.actions(inspectorPage).switchToHistory();
      await ui.historyTimeline.assertions(inspectorPage).timelineIsVisible();
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemIsVisible("say hello");
      await ui.historyTimeline
        .assertions(inspectorPage)
        .itemIsHidden("check account balance");
    });
  });
});
