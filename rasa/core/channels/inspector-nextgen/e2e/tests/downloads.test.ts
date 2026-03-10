import { test } from "@playwright/test";
import * as flows from "../flows/index";
import * as actions from "../actions/index";

test.describe("Download popover", () => {
  test.beforeEach(async ({ page }) => {
    await flows.inspector.navigateToInspectPageAndAssert(page);
  });

  test("Download button is visible on the page", async ({ page }) => {
    await actions.inspector.assertions(page).assertDownloadButtonVisible();
  });

  test("Download button is enabled after session start", async ({ page }) => {
    await flows.inspector.assertDownloadButtonEnabled(page);
  });

  test("Clicking download button opens popover with E2E and Conversation options", async ({
    page,
  }) => {
    await flows.inspector.openDownloadPopoverAndAssert(page);
  });

  test("Download E2E tests triggers a file download", async ({ page }) => {
    await flows.inspector.openDownloadPopoverAndAssert(page);

    const downloadPromise = page.waitForEvent("download");
    await actions.inspector.actions(page).clickDownloadE2e();
    const download = await downloadPromise;

    await test.step("Assert downloaded file is a YAML file", () => {
      const filename = download.suggestedFilename();
      test.expect(filename).toMatch(/^e2e-test-.*\.yml$/);
    });
  });

  test("Download Conversation triggers a file download", async ({ page }) => {
    await flows.inspector.openDownloadPopoverAndAssert(page);

    const downloadPromise = page.waitForEvent("download");
    await actions.inspector.actions(page).clickDownloadConversation();
    const download = await downloadPromise;

    await test.step("Assert downloaded file is a JSON file", () => {
      const filename = download.suggestedFilename();
      test.expect(filename).toMatch(/^conversation-.*\.json$/);
    });
  });
});
