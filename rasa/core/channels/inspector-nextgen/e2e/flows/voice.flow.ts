import { type Page, test } from "@playwright/test";

import * as ui from "@e2e/ui-actions";

export const typeTextAndAssertSendButtonVisible = async (
  page: Page,
  text: string,
) => {
  await test.step("Type text in message input", async () => {
    await ui.voiceControls.actions(page).typeMessage(text);
  });
  await test.step("Assert send button visible and voice button hidden", async () => {
    await ui.voiceControls.assertions(page).sendMessageButtonIsVisible();
    await ui.voiceControls.assertions(page).voiceStartButtonIsHidden();
  });
};

export const clearTextAndAssertVoiceButtonVisible = async (page: Page) => {
  await test.step("Clear message input", async () => {
    await ui.voiceControls.actions(page).clearMessage();
  });
  await test.step("Assert voice button visible and send button hidden", async () => {
    await ui.voiceControls.assertions(page).voiceStartButtonIsVisible();
    await ui.voiceControls.assertions(page).sendMessageButtonIsHidden();
  });
};

export const startVoiceCallAndAssertActive = async (page: Page) => {
  await test.step("Click start voice call", async () => {
    await ui.voiceControls.actions(page).startVoiceCall();
  });
  await test.step("Assert connecting or active state", async () => {
    await ui.voiceControls.assertions(page).connectingOrActiveStateIsVisible();
  });
  await test.step("Assert active voice call state", async () => {
    await ui.voiceControls.assertions(page).activeStateIsVisible();
    await ui.voiceControls.assertions(page).voiceStopButtonIsVisible();
  });
};

export const stopVoiceCallAndAssertInactive = async (page: Page) => {
  await test.step("Click stop voice call", async () => {
    await ui.voiceControls.actions(page).stopVoiceCall();
  });
  await test.step("Assert inactive voice state", async () => {
    await ui.voiceControls.assertions(page).inactiveStateIsVisible();
    await ui.voiceControls.assertions(page).voiceStartButtonIsVisible();
  });
};
