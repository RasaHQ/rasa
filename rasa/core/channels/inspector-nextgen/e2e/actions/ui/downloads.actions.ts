import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => ({
  button: page.getByTestId("download-button"),
  e2eOption: page.getByTestId("download-e2e"),
  conversationOption: page.getByTestId("download-conversation"),
});

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    openPopover: async () => {
      await locators.button.click();
    },
    clickE2eOption: async () => {
      await locators.e2eOption.click();
    },
    clickConversationOption: async () => {
      await locators.conversationOption.click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    buttonIsVisible: async () => {
      await expect(
        locators.button,
        "Download button should be visible",
      ).toBeVisible();
    },
    buttonIsDisabled: async () => {
      await expect(
        locators.button,
        "Download button should be disabled",
      ).toBeDisabled();
    },
    buttonIsEnabled: async () => {
      await expect(
        locators.button,
        "Download button should be enabled",
      ).toBeEnabled();
    },
    popoverIsVisible: async () => {
      await expect(
        locators.e2eOption,
        "Download E2E option should be visible",
      ).toBeVisible();
      await expect(
        locators.conversationOption,
        "Download Conversation option should be visible",
      ).toBeVisible();
    },
    popoverIsHidden: async () => {
      await expect(
        locators.e2eOption,
        "Download E2E option should not be visible",
      ).toBeHidden();
      await expect(
        locators.conversationOption,
        "Download Conversation option should not be visible",
      ).toBeHidden();
    },
  };
};
