import { type Page, test } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

type EventDetailsExpectation = {
  panelTitle: string;
  slotOrFlowName?: string;
  containsTexts?: string[];
  accordionTitles?: string[];
  actionEventInfoVisible?: boolean;
};

type MessageEventDetailsExpectation = Omit<EventDetailsExpectation, "panelTitle"> & {
  index?: number;
};

const assertEventDetailsState = async (
  page: Page,
  expectations: EventDetailsExpectation,
) => {
  await ui.eventDetails.assertions(page).panelIsVisible(expectations.panelTitle);

  if (expectations.slotOrFlowName) {
    await ui.eventDetails
      .assertions(page)
      .slotOrFlowNameIsVisible(expectations.slotOrFlowName);
  }

  for (const text of expectations.containsTexts ?? []) {
    await ui.eventDetails.assertions(page).panelContainsText(text);
  }

  if (expectations.actionEventInfoVisible) {
    await ui.eventDetails.assertions(page).actionEventInfoIsVisible();
  }

  for (const title of expectations.accordionTitles ?? []) {
    await ui.eventDetails.assertions(page).accordionIsVisible(title);
  }
};

export const sendMessageAndAssert = async (page: Page, message: string) => {
  await test.step("Send message", async () => {
    await ui.inspectorShell.actions(page).sendMessage(message);
  });
  await test.step("Assert user message in chat and input cleared", async () => {
    await ui.conversationLog.assertions(page).userMessageIsVisible(message);
    await ui.inspectorShell.assertions(page).inputIsCleared();
  });
};

export const sendMessageAndAssertBotReplies = async (
  page: Page,
  message: string,
  options?: { expectedBotMessageCount?: number },
) => {
  const botCountBefore = await ui.conversationLog
    .getLocators(page)
    .botMessages.count();
  await sendMessageAndAssert(page, message);
  await test.step("Assert bot response appears in chat", async () => {
    const assertions = ui.conversationLog.assertions(page);
    if (options?.expectedBotMessageCount === undefined) {
      await assertions.botMessageCountAtLeast(botCountBefore + 1);
    } else {
      await assertions.botMessageCountReached(options.expectedBotMessageCount);
    }
  });
};

export const sendMessageWithEnterAndAssert = async (
  page: Page,
  message: string,
) => {
  const userCountBefore = await ui.conversationLog
    .getLocators(page)
    .userMessages.count();
  await test.step("Send message with Enter", async () => {
    await ui.inspectorShell.actions(page).sendMessageWithEnter(message);
  });
  await test.step("Assert user message in chat and input cleared", async () => {
    await ui.conversationLog.assertions(page).userMessageIsVisible(message);
    await ui.inspectorShell.assertions(page).inputIsCleared();
    await ui.conversationLog
      .assertions(page)
      .userMessageCountIs(userCountBefore + 1);
  });
};

export const openConversationEventDetailsAndAssert = async (
  page: Page,
  expectations: EventDetailsExpectation & { eventName: string },
) => {
  await test.step(`Click event "${expectations.eventName}"`, async () => {
    await ui.conversationLog.actions(page).clickConversationEvent(
      expectations.eventName,
    );
  });
  await test.step(
    `Assert event details panel shows "${expectations.panelTitle}"`,
    async () => {
      await assertEventDetailsState(page, expectations);
    },
  );
};

export const openBotMessageDetailsAndAssert = async (
  page: Page,
  expectations: MessageEventDetailsExpectation,
) => {
  const { index = 0, ...panelExpectations } = expectations;

  await test.step(`Click bot message at index ${index}`, async () => {
    await ui.conversationLog.actions(page).clickBotMessage(index);
  });
  await test.step("Assert bot message details panel", async () => {
    await assertEventDetailsState(page, {
      panelTitle: "Agent response details",
      ...panelExpectations,
    });
  });
};

export const openUserMessageDetailsAndAssert = async (
  page: Page,
  expectations: MessageEventDetailsExpectation,
) => {
  const { index = 0, ...panelExpectations } = expectations;

  await test.step(`Click user message at index ${index}`, async () => {
    await ui.conversationLog.actions(page).clickUserMessage(index);
  });
  await test.step("Assert user message details panel", async () => {
    await assertEventDetailsState(page, {
      panelTitle: "User message details",
      ...panelExpectations,
    });
  });
};

export const closeEventDetailsAndAssertHidden = async (page: Page) => {
  await test.step("Close event details panel", async () => {
    await ui.eventDetails.actions(page).close();
  });
  await test.step("Assert event details panel is hidden", async () => {
    await ui.eventDetails.assertions(page).panelIsHidden();
  });
};

export const closeEventDetailsAndAssertCanvasRestored = async (page: Page) => {
  await closeEventDetailsAndAssertHidden(page);
  await test.step("Assert flow canvas is restored", async () => {
    await ui.inspectorCanvas.assertions(page).flowCanvasIsVisible();
  });
};

export const deselectConversationEventAndAssertCanvasRestored = async (
  page: Page,
  eventName: string,
) => {
  await test.step(`Click selected event "${eventName}" again`, async () => {
    await ui.conversationLog.actions(page).clickConversationEvent(eventName);
  });
  await test.step("Assert event details closes and canvas is restored", async () => {
    await ui.eventDetails.assertions(page).panelIsHidden();
    await ui.inspectorCanvas.assertions(page).flowCanvasIsVisible();
  });
};

export const clickBotMessageAndAssertNoEventDetails = async (
  page: Page,
  index = 0,
) => {
  await test.step(`Click bot message at index ${index}`, async () => {
    await ui.conversationLog.actions(page).clickBotMessage(index);
  });
  await test.step("Assert event details panel stays hidden", async () => {
    await ui.eventDetails.assertions(page).panelIsHidden();
  });
};
