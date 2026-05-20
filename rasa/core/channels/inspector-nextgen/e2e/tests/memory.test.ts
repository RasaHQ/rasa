import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

test.describe("Memory view", () => {
  test.use({
    inspectorOptions: {
      inspectMode: true,
    },
  });

  test("Switching to Memory view shows collected slots", async ({
    inspectorPage,
  }) => {
    await ui.viewMenu.actions(inspectorPage).switchToMemory();
    await ui.memoryPanel.assertions(inspectorPage).panelIsVisible();

    await test.step('Assert System slot "flow_hashes" is collected', async () => {
      await ui.memoryPanel.assertions(inspectorPage).sectionIsVisible("System");
      await ui.memoryPanel
        .assertions(inspectorPage)
        .slotIsVisible("System", "flow_hashes");
    });

    await test.step('Assert click "flow_hashes" opens event details', async () => {
      await flows.inspectPanel.clickCollectedSlot(
        inspectorPage,
        "System",
        "flow_hashes",
      );
      await ui.eventDetails.assertions(inspectorPage).panelIsVisible("Slot event");
      await ui.eventDetails
        .assertions(inspectorPage)
        .slotOrFlowNameIsVisible("flow_hashes");
      await ui.eventDetails.assertions(inspectorPage).slotValueIsVisible();
    });

    await test.step("Assert click x closes event details", async () => {
      await flows.chat.closeEventDetailsAndAssertHidden(inspectorPage);
      await ui.memoryPanel.assertions(inspectorPage).panelIsVisible();
    });
  });

  test("Slots are collected from session and current flow", async ({
    inspectorPage,
  }) => {
    await flows.chat.sendMessageAndAssertBotReplies(
      inspectorPage,
      "Add @tester",
      { expectedBotMessageCount: 2 },
    );
    await ui.viewMenu.actions(inspectorPage).switchToMemory();
    await ui.memoryPanel.assertions(inspectorPage).panelIsVisible();

    await test.step('Assert Session slot "contact handle" is collected', async () => {
      await ui.memoryPanel
        .assertions(inspectorPage)
        .sectionIsVisible("Session");
      await ui.memoryPanel
        .assertions(inspectorPage)
        .slotIsVisible("Session", "add_contact_handle", "@tester");
    });

    await flows.chat.sendMessageAndAssert(inspectorPage, "Tester A");
    await ui.conversationLog.assertions(inspectorPage).botMessageCountIs(2);

    await test.step('Assert Current flow slot "contact name" is collected', async () => {
      await ui.memoryPanel
        .assertions(inspectorPage)
        .sectionIsVisible("Current flow");
      await ui.memoryPanel
        .assertions(inspectorPage)
        .slotIsVisible("Current flow", "add_contact_name", "Tester A");
    });

    await test.step('Assert click "contact name" opens event details', async () => {
      await flows.inspectPanel.clickCollectedSlot(
        inspectorPage,
        "Current flow",
        "add_contact_name",
      );
      await ui.eventDetails.assertions(inspectorPage).panelIsVisible("Slot event");
      await ui.eventDetails
        .assertions(inspectorPage)
        .slotOrFlowNameIsVisible("add_contact_name");
      await ui.eventDetails.assertions(inspectorPage).slotValueIsVisible();
    });

    await test.step("Assert click x closes event details", async () => {
      await flows.chat.closeEventDetailsAndAssertHidden(inspectorPage);
      await ui.memoryPanel.assertions(inspectorPage).panelIsVisible();
    });

    await flows.chat.sendMessageAndAssert(inspectorPage, "Yes");
    await ui.conversationLog.assertions(inspectorPage).botMessageCountIs(5);

    await test.step("Assert slots are cleared on flow completion", async () => {
      await ui.memoryPanel
        .assertions(inspectorPage)
        .slotIsHidden("Session", "add_contact_handle");
      await ui.memoryPanel
        .assertions(inspectorPage)
        .slotIsHidden("Current flow", "add_contact_name");
    });
  });
});
