import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => {
  const inputField = page.getByTestId("assistant-input").locator("input");

  return {
    inputField,
    sendMessageButton: page.getByRole("button", { name: "Send message" }),
    voiceStartButton: page.getByRole("button", {
      name: "Start voice conversation",
    }),
    voiceStopButton: page.getByRole("button", {
      name: "Stop voice conversation",
    }),
  };
};

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    startVoiceCall: async () => {
      await locators.voiceStartButton.click();
    },
    stopVoiceCall: async () => {
      await locators.voiceStopButton.click();
    },
    typeMessage: async (text: string) => {
      await locators.inputField.fill(text);
    },
    clearMessage: async () => {
      await locators.inputField.fill("");
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    voiceStartButtonIsVisible: async () => {
      await expect(
        locators.voiceStartButton,
        "Voice start button should be visible",
      ).toBeVisible();
    },
    voiceStartButtonIsHidden: async () => {
      await expect(
        locators.voiceStartButton,
        "Voice start button should be hidden",
      ).toBeHidden();
    },
    voiceStopButtonIsVisible: async () => {
      await expect(
        locators.voiceStopButton,
        "Voice stop button should be visible",
      ).toBeVisible();
    },
    sendMessageButtonIsVisible: async () => {
      await expect(
        locators.sendMessageButton,
        "Send message button should be visible",
      ).toBeVisible();
    },
    sendMessageButtonIsHidden: async () => {
      await expect(
        locators.sendMessageButton,
        "Send message button should be hidden",
      ).toBeHidden();
    },
    connectingOrActiveStateIsVisible: async () => {
      await expect(
        locators.inputField,
        "Input placeholder should show 'Connecting...' or voice call in progress",
      ).toHaveAttribute(
        "placeholder",
        /Connecting\.\.\.|Voice conversation in progress \(\d{2}:\d{2}\)/,
      );
      await expect(
        locators.inputField,
        "Input should be disabled during connecting or active voice state",
      ).toBeDisabled();
    },
    activeStateIsVisible: async () => {
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
    inactiveStateIsVisible: async () => {
      await expect(
        locators.inputField,
        "Input placeholder should show 'Type your message'",
      ).toHaveAttribute("placeholder", "Type your message");
      await expect(
        locators.inputField,
        "Input should be disabled after a voice call ends",
      ).toBeDisabled();
    },
    timerHasIncremented: async () => {
      const initialPlaceholder = await locators.inputField.getAttribute(
        "placeholder",
      );
      const timerPattern = /Voice conversation in progress \((\d{2}):(\d{2})\)/;

      await expect
        .poll(
          async () => {
            const currentPlaceholder =
              (await locators.inputField.getAttribute("placeholder")) ?? "";
            const inTimerState = timerPattern.test(currentPlaceholder);
            const changed = currentPlaceholder !== initialPlaceholder;
            return inTimerState && changed ? currentPlaceholder : null;
          },
          {
            message:
              `Voice timer did not increment or voice call ended unexpectedly. Initial placeholder: "${initialPlaceholder}"`,
            timeout: 10000,
          },
        )
        .not.toBeNull();
    },
  };
};
