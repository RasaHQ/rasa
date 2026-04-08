import { expect, type Page } from "@playwright/test";

const INSPECT_PAGE_PATH = "/webhooks/inspector/inspect.html";

const getLocators = (page: Page) => {
  const inspectorCanvas = page.getByTestId("inspector-canvas");
  const conversationEvents = page.getByTestId("conversation-events");
  const assistantInput = page.getByTestId("assistant-input");
  const collectedSlotsSection = (section: string) => inspectorCanvas.getByTestId(`${section}-section`);
  return {
    inspectToggle: page.getByTestId("inspect-toggle"),
    viewControlInspect: page.getByTestId("view-control").getByText("Inspect"),
    viewControlChat: page.getByTestId("view-control").getByText("Chat"),
    restartConversation: page.getByTestId("restart-conversation"),
    tryAssistantContainer: page.getByTestId("try-assistant-container"),
    assistantInput,
    messageInputField: page.getByPlaceholder("Type your message"),
    inputField: assistantInput.locator("input"),
    sendMessageButton: page.getByRole("button", { name: "Send message" }),
    voiceStartButton: page.getByRole("button", {
      name: "Start voice conversation",
    }),
    voiceStopButton: page.getByRole("button", {
      name: "Stop voice conversation",
    }),
    assistantChat: page.getByTestId("assistant-chat"),
    loadingSpinner: page.getByTestId("loading-spinner"),
    inspectorCanvas,
    canvas: page.getByTestId("canvas"),
    flowNodes: page.getByTestId("node"),
    flowNode: (nodeName: string) =>
      inspectorCanvas.getByTestId("node").filter({ hasText: nodeName }),
    loadingDots: page.getByTestId("loading-dots"),
    userMessage: page.getByTestId("assistant-user-message"),
    botMessage: page.getByTestId("assistant-response"),
    conversationEvents,
    conversationEvent: (eventName: string) =>
      conversationEvents.getByText(eventName, { exact: true }),
    flowName: (flowName: string) => inspectorCanvas.getByText(flowName),
    eventDetailsClose: page.getByTestId("event-details-close"),
    eventSlotOrFlowName: page.getByTestId("event-slot-or-flow-name"),
    eventSlotValue: page.getByTestId("event-slot-value"),
    actionEventInfo: page.getByTestId("action-event-info"),
    downloadButton: page.getByTestId("download-button"),
    downloadE2e: page.getByTestId("download-e2e"),
    downloadConversation: page.getByTestId("download-conversation"),
    viewMenuButton: page.getByTestId("show-button"),
    viewMenuActiveFlow: page.getByTestId("view-menu-active-flow"),
    viewMenuHistory: page.getByTestId("view-menu-flow-history"),
    viewMenuMemory: page.getByTestId("view-menu-memory"),
    flowTimeline: inspectorCanvas.getByTestId("flow-timeline"),
    flowTimelineItem: inspectorCanvas.getByTestId("flow-timeline-item"),
    flowTimelineItemByName: (name: string) =>
      inspectorCanvas.getByTestId("flow-timeline-item").filter({ hasText: name }),
    historyPlaceholder: inspectorCanvas.getByText(
      "Conversation history will be shown here.",
    ),
    collectedSlot: (section: string, slotName: string) => collectedSlotsSection(section).getByTestId(`slot-${slotName}`),
  };
};

export const locators = getLocators;

export const actions = (page: Page) => {
  const locators = getLocators(page);
  return {
    navigateToInspectPage: async (query?: string) => {
      const path = query ? `${INSPECT_PAGE_PATH}${query}` : INSPECT_PAGE_PATH;
      await page.goto(path);
    },
    toggleInspect: async () => {
      await locators.inspectToggle.click();
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
      await locators.sendMessageButton.click();
    },
    sendMessageWithEnter: async (message: string) => {
      await locators.messageInputField.fill(message);
      await locators.messageInputField.press("Enter");
    },
    getUserMessageCount: async () => {
      return await locators.userMessage.count();
    },
    getBotMessageCount: async () => {
      return await locators.botMessage.count();
    },
    startVoiceCall: async () => {
      await locators.voiceStartButton.click();
    },
    stopVoiceCall: async () => {
      await locators.voiceStopButton.click();
    },
    typeInMessageInput: async (text: string) => {
      await locators.inputField.fill(text);
    },
    clearMessageInput: async () => {
      await locators.inputField.fill("");
    },
    clickConversationEvent: async (eventName: string) => {
      await locators.conversationEvent(eventName).click();
    },
    clickBotMessage: async (index = 0) => {
      await locators.botMessage.nth(index).click();
    },
    clickUserMessage: async (index = 0) => {
      await locators.userMessage.nth(index).click();
    },
    closeEventDetails: async () => {
      await locators.eventDetailsClose.click();
    },
    openDownloadPopover: async () => {
      await locators.downloadButton.click();
    },
    clickDownloadE2e: async () => {
      await locators.downloadE2e.click();
    },
    clickDownloadConversation: async () => {
      await locators.downloadConversation.click();
    },
    openViewMenu: async () => {
      await locators.viewMenuButton.click();
    },
    switchToHistoryView: async () => {
      await locators.viewMenuButton.click();
      await locators.viewMenuHistory.click();
    },
    switchToActiveFlowView: async () => {
      await locators.viewMenuButton.click();
      await locators.viewMenuActiveFlow.click();
    },
    switchToMemoryView: async () => {
      await locators.viewMenuButton.click();
      await locators.viewMenuMemory.click();
    },
    clickCollectedSlot: async (section: string, slotName: string) => {
      await locators.collectedSlot(section, slotName).click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);
  return {
    assertInspectPageLoaded: async () => {
      await expect(
        locators.viewControlInspect,
        "Inspector UI (view control) should be visible",
      ).toBeVisible();
    },
    assertHomePage: async () => {
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
    assertInspectorCanvasVisible: async () => {
      await expect(
        locators.inspectorCanvas,
        "Inspector canvas should be visible when Inspect is on",
      ).toBeVisible();
    },
    assertInspectorCanvasHidden: async () => {
      await expect(
        locators.inspectorCanvas,
        "Inspector canvas should not be visible when Inspect is off",
      ).toBeHidden();
    },
    assertFlowPanelNoActiveFlow: async () => {
      await expect(
        page
          .getByText("No flow is currently active")
          .or(page.getByText("Failed to load flows")),
        "Flow panel should show no active flow placeholder or flow load error",
      ).toBeVisible();
    },
    assertFlowPanelLoading: async () => {
      await expect(
        locators.loadingSpinner,
        "Loading spinner should be visible when loading flows",
      ).toBeVisible({ timeout: 15000 });
    },
    assertInputCleared: async () => {
      await expect(
        locators.messageInputField,
        "Message input should be empty after send",
      ).toHaveValue("");
    },
    assertUserMessageInChat: async (message: string) => {
      await expect(
        locators.userMessage.filter({ hasText: message }),
        "User message should appear in chat",
      ).toBeVisible();
    },
    assertCanvasWithNodesVisible: async () => {
      await expect(
        locators.canvas,
        "Flow canvas should be visible",
      ).toBeVisible();
      await expect(
        locators.flowNodes.first(),
        "At least one flow node should be visible",
      ).toBeVisible({ timeout: 10000 });
    },
    assertUserMessageCount: async (count: number) => {
      await expect(
        locators.userMessage,
        "User message count should be " + count,
      ).toHaveCount(count);
    },
    assertBotMessageCount: async (count: number) => {
      await expect(
        locators.botMessage,
        "Bot message count should be " + count,
      ).toHaveCount(count);
    },
    assertFlowNodeVisible: async (nodeName: string) => {
      await expect(
        locators.flowNode(nodeName),
        `Flow node "${nodeName}" should be visible`,
      ).toBeVisible({ timeout: 10000 });
    },
    assertConversationEventVisible: async (eventName: string) => {
      await expect(
        locators.conversationEvent(eventName),
        "Conversation event should be visible",
      ).toBeVisible();
    },
    assertActiveFlowName: async (flowName: string) => {
      await expect(
        locators.flowName(flowName),
        "Active flow name should be visible",
      ).toBeVisible();
    },
    assertBotResponse: async (response: string) => {
      await expect(
        locators.botMessage.filter({ hasText: response }),
        "Bot response should be visible",
      ).toBeVisible();
    },
    assertEventDetailsPanelVisible: async (title: string) => {
      await expect(
        page.getByRole("heading", { name: title }),
        `Event details panel with title "${title}" should be visible`,
      ).toBeVisible();
    },
    assertEventDetailsPanelHidden: async () => {
      await expect(
        locators.eventDetailsClose,
        "Event details close button should not be visible",
      ).toBeHidden();
    },
    assertEventSlotOrFlowName: async (name: string) => {
      await expect(
        locators.eventSlotOrFlowName,
        `Slot or flow name "${name}" should be visible`,
      ).toContainText(name);
    },
    assertEventSlotValue: async (value: string) => {
      await expect(
        locators.eventSlotValue,
        `Slot value "${value}" should be visible`,
      ).toContainText(value);
    },
    assertActionEventInfo: async () => {
      await expect(
        locators.actionEventInfo,
        "Action event info should be visible",
      ).toBeVisible();
    },
    assertDownloadButtonVisible: async () => {
      await expect(
        locators.downloadButton,
        "Download button should be visible",
      ).toBeVisible();
    },
    assertDownloadButtonDisabled: async () => {
      await expect(
        locators.downloadButton,
        "Download button should be disabled",
      ).toBeDisabled();
    },
    assertDownloadButtonEnabled: async () => {
      await expect(
        locators.downloadButton,
        "Download button should be enabled",
      ).toBeEnabled();
    },
    assertDownloadPopoverVisible: async () => {
      await expect(
        locators.downloadE2e,
        "Download E2E option should be visible",
      ).toBeVisible();
      await expect(
        locators.downloadConversation,
        "Download Conversation option should be visible",
      ).toBeVisible();
    },
    assertDownloadPopoverHidden: async () => {
      await expect(
        locators.downloadE2e,
        "Download E2E option should not be visible",
      ).toBeHidden();
    },
    assertFlowCanvasReplaced: async () => {
      await expect(
        locators.canvas,
        "Flow canvas should be hidden when event details is shown",
      ).toBeHidden();
    },
    assertFlowCanvasVisible: async () => {
      await expect(
        locators.canvas,
        "Flow canvas should be visible",
      ).toBeVisible();
    },
    assertHistoryPlaceholderVisible: async () => {
      await expect(
        locators.historyPlaceholder,
        "History placeholder should be visible when no flows have run",
      ).toBeVisible();
    },
    assertFlowTimelineVisible: async () => {
      await expect(
        locators.flowTimeline,
        "Flow timeline should be visible",
      ).toBeVisible({ timeout: 10000 });
    },
    assertCollectedSlotsVisible: async () => {
      await expect(
        locators.inspectorCanvas.getByRole("heading", { name: "Collected slots" }),
        "Collected slots panel should be visible in Memory view",
      ).toBeVisible();
    },
    assertCollectedSystemSlotsVisible: async () => {
      await expect(
        locators.inspectorCanvas.getByRole("heading", { name: "System" }),
        "System slots panel should be visible in Memory view",
      ).toBeVisible();
      await expect(
        locators.inspectorCanvas.getByTestId("System-section"),
        "System slots section should be visible in Memory view",
      ).toBeVisible();
    },
    assertCollectedSessionSlotsVisible: async () => {
      await expect(
        locators.inspectorCanvas.getByRole("heading", { name: "Session" }),
        "Session slots panel should be visible in Memory view",
      ).toBeVisible();
      await expect(
        locators.inspectorCanvas.getByTestId("Session-section"),
        "Session slots section should be visible in Memory view",
      ).toBeVisible();
    },
    assertCollectedCurrentFlowSlotsVisible: async () => {
      await expect(
        locators.inspectorCanvas.getByRole("heading", { name: "Current flow" }),
        "Current flow slots panel should be visible in Memory view",
      ).toBeVisible();
      await expect(
        locators.inspectorCanvas.getByTestId("Current flow-section"),
        "Current flow slots section should be visible in Memory view",
      ).toBeVisible();

    },
    assertCollectedSlotVisible: async (section: string, slotName: string, slotValue?: string) => {
      await expect(
        locators.collectedSlot(section, slotName),
        `Slot name "${slotName}" should be visible in Memory view`,
      ).toBeVisible();
      if (slotValue !== undefined) {
        await expect(
          locators.collectedSlot(section, slotName).getByTestId(`slot-value`),
          `Slot value "${slotValue}" should be visible in Memory view`,
        ).toContainText(slotValue);
      }
    },
    assertCollectedSlotNotVisible: async (section: string, slotName: string) => {
      await expect(
        locators.collectedSlot(section, slotName),
        `Slot name "${slotName}" should be hidden in Memory view`,
      ).toBeHidden();
    },
    assertFlowTimelineHidden: async () => {
      await expect(
        locators.flowTimeline,
        "Flow timeline should not be visible",
      ).toBeHidden();
    },
    assertFlowTimelineItemCount: async (count: number) => {
      await expect(
        locators.flowTimelineItem,
        `Flow timeline should have ${count} entries`,
      ).toHaveCount(count, { timeout: 10000 });
    },
    assertFlowTimelineItemVisible: async (flowName: string) => {
      await expect(
        locators.flowTimelineItemByName(flowName),
        `Flow timeline entry "${flowName}" should be visible`,
      ).toBeVisible({ timeout: 10000 });
    },
    assertFlowTimelineItemNotVisible: async (flowName: string) => {
      await expect(
        locators.flowTimelineItemByName(flowName),
        `Flow timeline entry "${flowName}" should not be visible`,
      ).toBeHidden({ timeout: 10000 });
    },
    assertFlowTimelineItemHasStatus: async (
      flowName: string,
      status: string,
    ) => {
      await expect(
        locators.flowTimelineItemByName(flowName),
        `Flow timeline entry "${flowName}" should show status "${status}"`,
      ).toContainText(status);
    },
    assertEventDetailsPanelContainsText: async (text: string) => {
      await expect(
        locators.inspectorCanvas,
        `Event details panel should contain text "${text}"`,
      ).toContainText(text);
    },
    assertEventDetailsAccordion: async (title: string) => {
      await expect(
        locators.inspectorCanvas.getByText(title, { exact: true }),
        `Accordion item "${title}" should be visible`,
      ).toBeVisible();
    },
    assertVoiceStartButtonVisible: async () => {
      await expect(
        locators.voiceStartButton,
        "Voice start button should be visible",
      ).toBeVisible();
    },
    assertVoiceStartButtonHidden: async () => {
      await expect(
        locators.voiceStartButton,
        "Voice start button should be hidden",
      ).toBeHidden();
    },
    assertVoiceStopButtonVisible: async () => {
      await expect(
        locators.voiceStopButton,
        "Voice stop button should be visible",
      ).toBeVisible();
    },
    assertSendMessageButtonVisible: async () => {
      await expect(
        locators.sendMessageButton,
        "Send message button should be visible",
      ).toBeVisible();
    },
    assertSendMessageButtonHidden: async () => {
      await expect(
        locators.sendMessageButton,
        "Send message button should be hidden",
      ).toBeHidden();
    },
    assertVoiceConnectingOrActiveState: async () => {
      await expect(
        locators.inputField,
        "Input placeholder should show 'Connecting...' or voice call in progress",
      ).toHaveAttribute(
        "placeholder",
        /Connecting\.\.\.|Voice conversation in progress \(\d{2}:\d{2}\)/,
      );
      await expect(
        locators.inputField,
        "Input should be disabled during connecting or active state",
      ).toBeDisabled();
    },
    assertVoiceActiveState: async () => {
      await expect(
        locators.inputField,
        "Input placeholder should show voice call in progress",
      ).toHaveAttribute(
        "placeholder",
        /Voice conversation in progress \(\d{2}:\d{2}\)/,
      );
      await expect(
        locators.inputField,
        "Input should be disabled during active voice call",
      ).toBeDisabled();
    },
    assertVoiceInactiveState: async () => {
      await expect(
        locators.inputField,
        "Input placeholder should show 'Type your message'",
      ).toHaveAttribute("placeholder", "Type your message");
      await expect(
        locators.inputField,
        "Input should be enabled when voice is inactive",
      ).toBeEnabled();
    },
    assertSlotEventVisible: async (slotName: string) => {
      await expect(
        locators.inspectorCanvas.getByRole("heading", { name: "Slot event" }),
        "Slot event heading should be visible",
      ).toBeVisible();
      await expect(
        locators.eventSlotOrFlowName,
        `Slot "${slotName}" should be visible`,
      ).toContainText(slotName);
      await expect(
        locators.eventSlotValue,
        "Slot value should be visible",
      ).toBeVisible();
    },
  };
};
