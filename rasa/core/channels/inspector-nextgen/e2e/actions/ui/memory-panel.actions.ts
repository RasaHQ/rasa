import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => {
  const inspectorCanvas = page.getByTestId("inspector-canvas");
  const section = (name: string) => inspectorCanvas.getByTestId(`${name}-section`);

  return {
    inspectorCanvas,
    panelHeading: inspectorCanvas.getByRole("heading", { name: "Collected slots" }),
    sectionHeading: (name: string) =>
      inspectorCanvas.getByRole("heading", { name }),
    section,
    slot: (sectionName: string, slotName: string) =>
      section(sectionName).getByTestId(`slot-${slotName}`),
    slotValue: (sectionName: string, slotName: string) =>
      section(sectionName)
        .getByTestId(`slot-${slotName}`)
        .getByTestId("slot-value"),
  };
};

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    clickSlot: async (sectionName: string, slotName: string) => {
      await locators.slot(sectionName, slotName).click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    panelIsVisible: async () => {
      await expect(
        locators.panelHeading,
        "Collected slots panel should be visible in Memory view",
      ).toBeVisible();
    },
    sectionIsVisible: async (sectionName: string) => {
      await expect(
        locators.sectionHeading(sectionName),
        `${sectionName} slots heading should be visible in Memory view`,
      ).toBeVisible();
      await expect(
        locators.section(sectionName),
        `${sectionName} slots section should be visible in Memory view`,
      ).toBeVisible();
    },
    slotIsVisible: async (
      sectionName: string,
      slotName: string,
      slotValue?: string,
    ) => {
      await expect(
        locators.slot(sectionName, slotName),
        `Slot "${slotName}" should be visible in the ${sectionName} section`,
      ).toBeVisible();

      if (slotValue !== undefined) {
        await expect(
          locators.slotValue(sectionName, slotName),
          `Slot "${slotName}" should show value "${slotValue}" in the ${sectionName} section`,
        ).toContainText(slotValue);
      }
    },
    slotIsHidden: async (sectionName: string, slotName: string) => {
      await expect(
        locators.slot(sectionName, slotName),
        `Slot "${slotName}" should be hidden in the ${sectionName} section`,
      ).toBeHidden();
    },
  };
};
