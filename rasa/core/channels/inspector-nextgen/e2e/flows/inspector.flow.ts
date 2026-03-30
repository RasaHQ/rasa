import { expect, test, type Page } from "@playwright/test";
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

export const assertVoiceButtonVisibleOnLoad = async (page: Page) => {
  await test.step("Assert voice start button visible on page load", async () => {
    await actions.inspector.assertions(page).assertVoiceStartButtonVisible();
  });
  await test.step("Assert send message button hidden on page load", async () => {
    await actions.inspector.assertions(page).assertSendMessageButtonHidden();
  });
};

export const typeTextAndAssertSendButtonVisible = async (
  page: Page,
  text: string,
) => {
  await test.step("Type text in message input", async () => {
    await actions.inspector.actions(page).typeInMessageInput(text);
  });
  await test.step("Assert send button visible and voice button hidden", async () => {
    await actions.inspector.assertions(page).assertSendMessageButtonVisible();
    await actions.inspector.assertions(page).assertVoiceStartButtonHidden();
  });
};

export const clearTextAndAssertVoiceButtonVisible = async (page: Page) => {
  await test.step("Clear message input", async () => {
    await actions.inspector.actions(page).clearMessageInput();
  });
  await test.step("Assert voice button visible and send button hidden", async () => {
    await actions.inspector.assertions(page).assertVoiceStartButtonVisible();
    await actions.inspector.assertions(page).assertSendMessageButtonHidden();
  });
};

export const startVoiceCallAndAssertActive = async (page: Page) => {
  await test.step("Click start voice call", async () => {
    await actions.inspector.actions(page).startVoiceCall();
  });
  await test.step("Assert connecting or active state", async () => {
    await actions.inspector.assertions(page).assertVoiceConnectingOrActiveState();
  });
  await test.step("Assert active voice call state", async () => {
    await actions.inspector.assertions(page).assertVoiceActiveState();
    await actions.inspector.assertions(page).assertVoiceStopButtonVisible();
  });
};

export const assertVoiceTimerIncremented = async (page: Page) => {
  await test.step("Assert voice call timer increments", async () => {
    const inputField = actions.inspector.locators(page).inputField;
    const initialPlaceholder = await inputField.getAttribute("placeholder");
    const timerPattern = /Voice conversation in progress \((\d{2}):(\d{2})\)/;

    await expect
      .poll(
        async () => {
          const current = await inputField.getAttribute("placeholder");
          const match = current?.match(timerPattern);
          return match ? current : null;
        },
        {
          message: `Timer did not increment or voice call ended unexpectedly. Initial: "${initialPlaceholder}"`,
          timeout: 10000,
        },
      )
      .not.toBe(initialPlaceholder);
  });
};

export const stopVoiceCallAndAssertInactive = async (page: Page) => {
  await test.step("Click stop voice call", async () => {
    await actions.inspector.actions(page).stopVoiceCall();
  });
  await test.step("Assert inactive voice state", async () => {
    await actions.inspector.assertions(page).assertVoiceInactiveState();
    await actions.inspector.assertions(page).assertVoiceStartButtonVisible();
  });
};

export const switchToHistoryView = async (page: Page) => {
  await test.step("Switch to History view", async () => {
    await actions.inspector.actions(page).switchToHistoryView();
  });
};

export const switchToActiveFlowView = async (page: Page) => {
  await test.step("Switch to Active Flow view", async () => {
    await actions.inspector.actions(page).switchToActiveFlowView();
  });
};

export const assertHistoryPlaceholder = async (page: Page) => {
  await test.step("Assert history placeholder is visible", async () => {
    await actions.inspector
      .assertions(page)
      .assertHistoryPlaceholderVisible();
  });
};

export const assertFlowTimelineVisible = async (page: Page) => {
  await test.step("Assert flow timeline is visible", async () => {
    await actions.inspector.assertions(page).assertFlowTimelineVisible();
  });
};

export const assertFlowTimelineItemCount = async (
  page: Page,
  count: number,
) => {
  await test.step(`Assert flow timeline has ${count} entries`, async () => {
    await actions.inspector
      .assertions(page)
      .assertFlowTimelineItemCount(count);
  });
};

export const assertFlowTimelineItemVisible = async (
  page: Page,
  flowName: string,
) => {
  await test.step(`Assert flow "${flowName}" in timeline`, async () => {
    await actions.inspector
      .assertions(page)
      .assertFlowTimelineItemVisible(flowName);
  });
};

export const assertFlowTimelineItemNotVisible = async (
  page: Page,
  flowName: string,
) => {
  await test.step(`Assert flow "${flowName}" not in timeline`, async () => {
    await actions.inspector
      .assertions(page)
      .assertFlowTimelineItemNotVisible(flowName);
  });
};

export const assertFlowTimelineItemHasStatus = async (
  page: Page,
  flowName: string,
  status: string,
) => {
  await test.step(`Assert flow "${flowName}" has status "${status}"`, async () => {
    await actions.inspector
      .assertions(page)
      .assertFlowTimelineItemHasStatus(flowName, status);
  });
};
