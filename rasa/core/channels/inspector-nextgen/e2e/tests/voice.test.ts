import { test } from "@e2e/fixtures";
import * as flows from "@e2e/flows";
import * as ui from "@e2e/ui-actions";

test.describe("Voice functionality", () => {
  test("Voice and send button toggle based on input text", async ({
    inspectorPage,
  }) => {
    await ui.voiceControls.assertions(inspectorPage).voiceStartButtonIsVisible();
    await ui.voiceControls
      .assertions(inspectorPage)
      .sendMessageButtonIsHidden();
    await flows.voice.typeTextAndAssertSendButtonVisible(
      inspectorPage,
      "Hello",
    );
    await flows.voice.clearTextAndAssertVoiceButtonVisible(inspectorPage);
  });

  test("Voice call workflow: start, verify active state, stop", async ({
    inspectorPage,
  }) => {
    await flows.voice.startVoiceCallAndAssertActive(inspectorPage);
    await ui.voiceControls.assertions(inspectorPage).timerHasIncremented();
    await flows.voice.stopVoiceCallAndAssertInactive(inspectorPage);
  });
});
