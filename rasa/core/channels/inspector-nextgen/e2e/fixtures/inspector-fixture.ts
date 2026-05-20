import { test as base, type Page } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

export type InspectorOptions = {
  open?: boolean;
  inspectMode?: boolean;
  query?: string;
};

type InspectorFixtures = {
  inspectorOptions: InspectorOptions;
  inspectorPage: Page;
};

const defaultInspectorOptions: InspectorOptions = {
  open: true,
};

const openInspectorPage = async (page: Page, options: InspectorOptions) => {
  const resolvedOptions = {
    ...defaultInspectorOptions,
    ...options,
  };

  if (!resolvedOptions.open) {
    return;
  }

  await ui.inspectorShell.actions(page).navigateToInspectPage(resolvedOptions.query);
  await ui.inspectorShell.assertions(page).pageIsLoaded();
  await ui.inspectorShell.assertions(page).shellIsReady();

  if (resolvedOptions.inspectMode) {
    await ui.inspectorShell.actions(page).toggleInspectOn();
    await ui.inspectorCanvas.assertions(page).inspectorCanvasIsVisible();
  }
};

export const inspectorFixture = base.extend<InspectorFixtures>({
  inspectorOptions: [{ ...defaultInspectorOptions }, { option: true }],

  inspectorPage: async ({ page, inspectorOptions }, use) => {
    await openInspectorPage(page, inspectorOptions);
    await use(page);
  },
});
