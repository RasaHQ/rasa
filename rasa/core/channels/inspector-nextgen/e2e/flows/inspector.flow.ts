import { test, type Page } from "@playwright/test";
import * as actions from "../actions/index";

export const navigateToInspectPageAndAssert = async (
  page: Page,
  options?: { query?: string },
) => {
  await test.step("Navigate to inspect page", async () => {
    await actions.inspector.actions(page).navigateToInspectPage(options?.query);
  });
  await test.step("Assert inspector page loaded", async () => {
    await actions.inspector.assertions(page).assertInspectPageLoaded();
  });
  await test.step("Assert shell loaded (container, input, chat, no spinner)", async () => {
    await actions.inspector.assertions(page).assertHomePage();
  });
};

export const toggleInspectOnAndAssertFlowPanel = async (page: Page) => {
  await test.step("Turn Inspect on", async () => {
    await actions.inspector.actions(page).toggleInspect();
  });
  await test.step("Assert inspector canvas visible", async () => {
    await actions.inspector.assertions(page).assertInspectorCanvasVisible();
  });
};

export const toggleInspectOnAndAssertNoActiveFlow = async (page: Page) => {
  await test.step("Turn Inspect on", async () => {
    await actions.inspector.actions(page).toggleInspect();
  });
  await test.step("Assert flow panel shows no active flow placeholder", async () => {
    await actions.inspector.assertions(page).assertInspectorCanvasVisible();
    await actions.inspector.assertions(page).assertFlowPanelNoActiveFlow();
  });
};

export const assertFlowPanelShowsFlowWithDetails = async (page: Page) => {
  await test.step("Assert flow panel shows flow with details", async () => {
    await actions.inspector.assertions(page).assertCanvasWithNodesVisible();
  });
};

export const toggleInspectOffAndAssertCanvasHidden = async (page: Page) => {
  await test.step("Turn Inspect off", async () => {
    await actions.inspector.actions(page).toggleInspect();
  });
  await test.step("Assert inspector canvas hidden", async () => {
    await actions.inspector.assertions(page).assertInspectorCanvasHidden();
  });
};

export const restartConversationAndAssert = async (page: Page) => {
  await test.step("Click restart conversation", async () => {
    await actions.inspector.actions(page).restartConversation();
  });
  await test.step("Assert conversation area reset", async () => {
    const chat = actions.inspector.locators(page).assistantChat;
    await chat.waitFor({ state: "visible" });
  });
};

export const sendMessageAndAssert = async (page: Page, message: string) => {
  await test.step("Send message", async () => {
    await actions.inspector.actions(page).sendMessage(message);
  });
  await test.step("Assert user message in chat and input cleared", async () => {
    await actions.inspector.assertions(page).assertUserMessageInChat(message);
    await actions.inspector.assertions(page).assertInputCleared();
  });
};

export const sendMessageAndAssertBotReplies = async (
  page: Page,
  message: string,
) => {
  const botCountBefore = await actions.inspector
    .actions(page)
    .getBotMessageCount();
  await sendMessageAndAssert(page, message);
  await test.step("Assert bot response appears in chat", async () => {
    await actions.inspector
      .assertions(page)
      .assertBotMessageCount(botCountBefore + 1);
  });
};

export const sendMessageWithEnterAndAssert = async (
  page: Page,
  message: string,
) => {
  await test.step("Send message with Enter", async () => {
    await actions.inspector.actions(page).sendMessageWithEnter(message);
  });
  await test.step("Assert user message in chat and input cleared", async () => {
    await actions.inspector.assertions(page).assertUserMessageInChat(message);
    await actions.inspector.assertions(page).assertInputCleared();
  });
};

export const assertActiveFlowName = async (page: Page, flowName: string) => {
  await test.step("Assert active flow name", async () => {
    await actions.inspector.assertions(page).assertActiveFlowName(flowName);
  });
};

export const assertConversationEventVisible = async (
  page: Page,
  eventName: string,
) => {
  await test.step(`Assert conversation event ${eventName} is visible`, async () => {
    await actions.inspector
      .assertions(page)
      .assertConversationEventVisible(eventName);
  });
};

export const assertFlowNodeVisible = async (page: Page, nodeName: string) => {
  await test.step(`Assert flow node "${nodeName}" is visible`, async () => {
    await actions.inspector.assertions(page).assertFlowNodeVisible(nodeName);
  });
};

export const assertBotResponse = async (page: Page, response: string) => {
  await test.step(`Assert bot response ${response} is visible`, async () => {
    await actions.inspector.assertions(page).assertBotResponse(response);
  });
};

export const clickConversationEventAndAssertPanel = async (
  page: Page,
  eventName: string,
  expectedTitle: string,
) => {
  await test.step(`Click event "${eventName}"`, async () => {
    await actions.inspector.actions(page).clickConversationEvent(eventName);
  });
  await test.step(`Assert event details panel shows "${expectedTitle}"`, async () => {
    await actions.inspector.assertions(page).assertEventDetailsPanelVisible(expectedTitle);
  });
};

export const closeEventDetailsAndAssertHidden = async (page: Page) => {
  await test.step("Close event details panel", async () => {
    await actions.inspector.actions(page).closeEventDetails();
  });
  await test.step("Assert event details panel is hidden", async () => {
    await actions.inspector.assertions(page).assertEventDetailsPanelHidden();
  });
};

export const assertFlowCanvasReplacedByEventDetails = async (page: Page) => {
  await test.step("Assert flow canvas replaced by event details", async () => {
    await actions.inspector.assertions(page).assertFlowCanvasReplaced();
  });
};

export const assertFlowCanvasRestored = async (page: Page) => {
  await test.step("Assert flow canvas is restored", async () => {
    await actions.inspector.assertions(page).assertFlowCanvasVisible();
  });
};

export const openDownloadPopoverAndAssert = async (page: Page) => {
  await test.step("Open download popover", async () => {
    await actions.inspector.actions(page).openDownloadPopover();
  });
  await test.step("Assert download options visible", async () => {
    await actions.inspector.assertions(page).assertDownloadPopoverVisible();
  });
};

export const assertDownloadButtonDisabled = async (page: Page) => {
  await test.step("Assert download button is disabled", async () => {
    await actions.inspector.assertions(page).assertDownloadButtonDisabled();
  });
};

export const assertDownloadButtonEnabled = async (page: Page) => {
  await test.step("Assert download button is enabled", async () => {
    await actions.inspector.assertions(page).assertDownloadButtonEnabled();
  });
};

export const clickBotMessageAndAssertPanel = async (
  page: Page,
  index = 0,
) => {
  await test.step(`Click bot message at index ${index}`, async () => {
    await actions.inspector.actions(page).clickBotMessage(index);
  });
  await test.step("Assert agent response details panel visible", async () => {
    await actions.inspector
      .assertions(page)
      .assertEventDetailsPanelVisible("Agent response details");
  });
};

export const clickUserMessageAndAssertPanel = async (
  page: Page,
  index = 0,
) => {
  await test.step(`Click user message at index ${index}`, async () => {
    await actions.inspector.actions(page).clickUserMessage(index);
  });
  await test.step("Assert user message details panel visible", async () => {
    await actions.inspector
      .assertions(page)
      .assertEventDetailsPanelVisible("User message details");
  });
};
