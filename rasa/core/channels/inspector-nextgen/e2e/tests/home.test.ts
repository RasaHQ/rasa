import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

test.describe("Inspector Landing page", () => {
  test("Inspect toggle on shows flow panel, off hides canvas", async ({
    inspectorPage,
  }) => {
    await flows.shell.openInspectModeAndAssertCanvas(inspectorPage);
    await flows.shell.closeInspectModeAndAssertCanvasHidden(inspectorPage);
  });

  test("Restart conversation clears user messages", async ({
    inspectorPage,
  }) => {
    await flows.chat.sendMessageAndAssert(inspectorPage, "Hi there!");
    await flows.shell.restartConversationAndAssertReset(inspectorPage);
  });

  test("Send message with Enter key adds user message and clears input", async ({
    inspectorPage,
  }) => {
    await flows.chat.sendMessageWithEnterAndAssert(inspectorPage, "Hello");
  });

  test.describe("open: false", () => {
    test.use({ inspectorOptions: { open: false } });

    test("loads when navigating manually", async ({ inspectorPage }) => {
      await ui.inspectorShell.actions(inspectorPage).navigateToInspectPage();
      await flows.shell.assertLandingShellReady(inspectorPage);
    });
  });

  test.describe("URL with token=None", () => {
    test.use({
      inspectorOptions: {
        query: "?token=None",
      },
    });

    test("loads without error", async ({ inspectorPage }) => {
      await flows.shell.assertLandingShellReady(inspectorPage);
    });
  });
});
