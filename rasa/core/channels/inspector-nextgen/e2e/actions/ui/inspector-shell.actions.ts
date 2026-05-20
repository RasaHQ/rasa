import { expect, type Page } from "@playwright/test";

const INSPECT_PAGE_PATH = "/webhooks/inspector/inspect.html";

export const getLocators = (page: Page) => {
  const assistantInput = page.getByTestId("assistant-input");

  return {
    viewControlInspect: page.getByTestId("view-control").getByText("Inspect"),
    viewControlChat: page.getByTestId("view-control").getByText("Chat"),
    restartConversation: page.getByTestId("restart-conversation"),
    tryAssistantContainer: page.getByTestId("try-assistant-container"),
    assistantInput,
    messageInputField: page.getByPlaceholder("Type your message"),
    assistantChat: page.getByTestId("assistant-chat"),
    loadingSpinner: page.getByTestId("loading-spinner"),
  };
};

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    navigateToInspectPage: async (query?: string) => {
      let suffix = "";
      if (query) {
        suffix = query.startsWith("?") ? query : `?${query}`;
      }
      await page.goto(`${INSPECT_PAGE_PATH}${suffix}`);
    },
    toggleInspectOn: async () => {
      await locators.viewControlInspect.click();
    },
    toggleInspectOff: async () => {
      await locators.viewControlChat.click();
    },
    restartConversation: async () => {
      await locators.restartConversation.click();
    },
    sendMessage: async (message: string) => {
      await locators.messageInputField.fill(message);
      await page.getByRole("button", { name: "Send message" }).click();
    },
    sendMessageWithEnter: async (message: string) => {
      await locators.messageInputField.fill(message);
      await locators.messageInputField.press("Enter");
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    pageIsLoaded: async () => {
      await expect(
        locators.viewControlInspect,
        "Inspector UI (view control) should be visible",
      ).toBeVisible();
    },
    shellIsReady: async () => {
      await expect(
        locators.tryAssistantContainer,
        "Try assistant container should be visible",
      ).toBeVisible();
      await expect(
        locators.assistantInput,
        "Message input should be visible",
      ).toBeVisible();
      await expect(
        locators.assistantChat,
        "Assistant chat area should be visible",
      ).toBeVisible();
      await expect(
        locators.loadingSpinner,
        "Loading spinner should not be visible after load",
      ).toBeHidden();
    },
    inputIsCleared: async () => {
      await expect(
        locators.messageInputField,
        "Message input should be empty after send",
      ).toHaveValue("");
    },
  };
};
