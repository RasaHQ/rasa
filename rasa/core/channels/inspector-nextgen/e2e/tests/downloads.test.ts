import { expect, test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

test.describe("Download popover", () => {
  test("Download button is visible on the page", async ({ inspectorPage }) => {
    await ui.downloads.assertions(inspectorPage).buttonIsVisible();
  });

  test("Download button is enabled after session start", async ({
    inspectorPage,
  }) => {
    await ui.downloads.assertions(inspectorPage).buttonIsEnabled();
  });

  test("Clicking download button opens popover with E2E and Conversation options", async ({
    inspectorPage,
  }) => {
    await flows.downloads.openDownloadPopoverAndAssert(inspectorPage);
  });

  test("Download E2E tests triggers a file download", async ({
    inspectorPage,
  }) => {
    await flows.downloads.openDownloadPopoverAndAssert(inspectorPage);
    const download = await flows.downloads.downloadE2eFile(inspectorPage);

    await test.step("Assert downloaded file is a YAML file", () => {
      const filename = download.suggestedFilename();
      expect(filename).toMatch(/^e2e-test-.*\.yml$/);
    });
  });

  test("Download Conversation triggers a file download", async ({
    inspectorPage,
  }) => {
    await flows.downloads.openDownloadPopoverAndAssert(inspectorPage);
    const download = await flows.downloads.downloadConversationFile(
      inspectorPage,
    );

    await test.step("Assert downloaded file is a JSON file", () => {
      const filename = download.suggestedFilename();
      expect(filename).toMatch(/^conversation-.*\.json$/);
    });
  });
});
