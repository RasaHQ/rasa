import { type Download, type Page, test } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

export const openDownloadPopoverAndAssert = async (page: Page) => {
  await test.step("Open download popover", async () => {
    await ui.downloads.actions(page).openPopover();
  });

  await test.step("Assert download options visible", async () => {
    await ui.downloads.assertions(page).popoverIsVisible();
  });
};

export const downloadE2eFile = async (page: Page): Promise<Download> => {
  return test.step("Download E2E test file", async () => {
    const downloadPromise = page.waitForEvent("download");
    await ui.downloads.actions(page).clickE2eOption();
    return downloadPromise;
  });
};

export const downloadConversationFile = async (
  page: Page,
): Promise<Download> => {
  return test.step("Download conversation file", async () => {
    const downloadPromise = page.waitForEvent("download");
    await ui.downloads.actions(page).clickConversationOption();
    return downloadPromise;
  });
};
