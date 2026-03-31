import { test } from "@playwright/test";
import * as flows from "../flows/index";
import * as actions from "../actions/index";

test.describe("Memory view", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
    await flows.inspector.toggleInspectOnAndAssertFlowPanel(page);
  });

  test("Switching to Memory view shows collected slots", async ({
    page,
  }) => {
    await flows.inspector.switchToMemoryView(page);
    await flows.inspector.assertCollectedSlotsVisible(page);

    await test.step("Assert System slot \"flow_hashes\" is collected", async () => {
      await flows.inspector.assertCollectedSystemSlotsVisible(page);
      await flows.inspector.assertCollectedSlotVisible(page, "System", "flow_hashes");
    });

    await test.step("Assert click \"flow_hashes\" opens event details", async () => {
      await flows.inspector.clickCollectedSlot(page, "System", "flow_hashes");
      await flows.inspector.assertSlotEventVisible(page, "flow_hashes");
    });

    await test.step("Assert click x closes event details", async () => {
      await flows.inspector.closeEventDetailsAndAssertHidden(page);
      await flows.inspector.assertCollectedSlotsVisible(page);
    });
  });

  test("Slots are collected from session and current flow", async ({
    page,
  }) => {
    await flows.inspector.sendMessageAndAssert(page, "Add @tester");
    await actions.inspector.assertions(page).assertBotMessageCount(2);
    await flows.inspector.switchToMemoryView(page);
    await flows.inspector.assertCollectedSlotsVisible(page);

    await test.step("Assert Session slot \"contact handle\" is collected", async () => {
      await flows.inspector.assertCollectedSessionSlotsVisible(page);
      await flows.inspector.assertCollectedSlotVisible(page, "Session", "add_contact_handle", "@tester");
    });

    await flows.inspector.sendMessageAndAssert(page, "Tester A");
    await actions.inspector.assertions(page).assertBotMessageCount(3);

    await test.step("Assert Current flow slot \"contact name\" is collected", async () => {
      await flows.inspector.assertCollectedCurrentFlowSlotsVisible(page);
      await flows.inspector.assertCollectedSlotVisible(page, "Current flow", "add_contact_name", "Tester A");
    });

    await test.step("Assert click \"contact name\" opens event details", async () => {
      await flows.inspector.clickCollectedSlot(page, "Current flow", "add_contact_name");
      await flows.inspector.assertSlotEventVisible(page, "add_contact_name");
    });

    await test.step("Assert click x closes event details", async () => {
      await flows.inspector.closeEventDetailsAndAssertHidden(page);
      await flows.inspector.assertCollectedSlotsVisible(page);
    });

    await flows.inspector.sendMessageAndAssert(page, "Yes");
    await actions.inspector.assertions(page).assertBotMessageCount(4);

    await test.step("Assert slots are cleared on flow completion", async () => {
      await flows.inspector.assertCollectedSlotNotVisible(page, "Session", "add_contact_handle");
      await flows.inspector.assertCollectedSlotNotVisible(page, "Current flow", "add_contact_name");
    });
  });
});
