import { type Page, test } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

export const clickCollectedSlot = async (
  page: Page,
  section: string,
  slotName: string,
) => {
  await test.step(`Assert collected slot "${slotName}" is visible`, async () => {
    await ui.memoryPanel.assertions(page).slotIsVisible(section, slotName);
  });

  await test.step(`Click collected slot "${slotName}"`, async () => {
    await ui.memoryPanel.actions(page).clickSlot(section, slotName);
  });
};
