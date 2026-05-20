import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => ({
  toggleButton: page.getByTestId("show-button"),
  activeFlowOption: page.getByTestId("view-menu-active-flow"),
  historyOption: page.getByTestId("view-menu-flow-history"),
  memoryOption: page.getByTestId("view-menu-memory"),
});

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    open: async () => {
      await locators.toggleButton.click();
    },
    switchToActiveFlow: async () => {
      await locators.toggleButton.click();
      await locators.activeFlowOption.click();
    },
    switchToHistory: async () => {
      await locators.toggleButton.click();
      await locators.historyOption.click();
    },
    switchToMemory: async () => {
      await locators.toggleButton.click();
      await locators.memoryOption.click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    toggleButtonIsVisible: async () => {
      await expect(
        locators.toggleButton,
        "Inspector view menu toggle should be visible",
      ).toBeVisible();
    },
    activeFlowOptionIsVisible: async () => {
      await expect(
        locators.activeFlowOption,
        "Active Flow option should be visible in the inspector view menu",
      ).toBeVisible();
    },
    historyOptionIsVisible: async () => {
      await expect(
        locators.historyOption,
        "History option should be visible in the inspector view menu",
      ).toBeVisible();
    },
    memoryOptionIsVisible: async () => {
      await expect(
        locators.memoryOption,
        "Memory option should be visible in the inspector view menu",
      ).toBeVisible();
    },
  };
};
