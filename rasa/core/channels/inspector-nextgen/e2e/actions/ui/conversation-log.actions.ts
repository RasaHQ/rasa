import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => {
  const chat = page.getByTestId("assistant-chat");
  // Match only non–last-active events. The last active event uses
  // `conversation-events-last-active-event` and often duplicates labels such as
  // "Waiting for user input" (action_listen) already shown earlier in the log.
  const conversationEvents = chat.getByTestId("conversation-events");

  return {
    chat,
    conversationEvents,
    conversationEvent: (eventName: string) =>
      conversationEvents.getByText(eventName, { exact: true }),
    userMessages: chat.getByTestId("assistant-user-message"),
    botMessages: chat.getByTestId("assistant-response"),
  };
};

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    clickConversationEvent: async (eventName: string) => {
      await locators.conversationEvent(eventName).click();
    },
    clickBotMessage: async (index = 0) => {
      await locators.botMessages.nth(index).click();
    },
    clickUserMessage: async (index = 0) => {
      await locators.userMessages.nth(index).click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    userMessageIsVisible: async (message: string) => {
      await expect(
        locators.userMessages.filter({ hasText: message }),
        `User message "${message}" should appear in chat`,
      ).toBeVisible();
    },
    userMessageCountIs: async (count: number) => {
      await expect(
        locators.userMessages,
        `User message count should be ${count}`,
      ).toHaveCount(count);
    },
    botMessageCountIs: async (count: number) => {
      await expect(
        locators.botMessages,
        `Bot message count should be ${count}`,
      ).toHaveCount(count);
    },
    botMessageCountReached: async (count: number) => {
      await expect
        .poll(
          async () => locators.botMessages.count(),
          {
            message: `Bot message count should reach ${count}`,
          },
        )
        .toBe(count);
    },
    botMessageCountAtLeast: async (minCount: number) => {
      await expect
        .poll(
          async () => (await locators.botMessages.count()) >= minCount,
          {
            message: `Bot message count should be at least ${minCount}`,
          },
        )
        .toBe(true);
    },
    conversationEventIsVisible: async (eventName: string) => {
      await expect(
        locators.conversationEvent(eventName),
        `Conversation event "${eventName}" should be visible`,
      ).toBeVisible();
    },
  };
};
