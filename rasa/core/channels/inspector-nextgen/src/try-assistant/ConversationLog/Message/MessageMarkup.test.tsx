import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/utils";
import { UtteranceType, type Utterance } from "../../../types";
import { MessageMarkup } from "./MessageMarkup";

describe("MessageMarkup", () => {
  it("hides the avatar for an empty bot utterance", () => {
    const emptyBotUtterance: Omit<Utterance, "entities"> = {
      __typename: "Utterance",
      id: "1",
      type: UtteranceType.Bot,
      text: "",
      timestamp: new Date().toISOString(),
      originalTimestamp: 0,
      tokens: [],
      rephrase: false,
      rephrasePrompt: null,
      metadata: { parseData: {} },
    };

    renderWithProviders(<MessageMarkup utterance={emptyBotUtterance} />);

    expect(screen.queryByTestId("message-avatar")).not.toBeInTheDocument();
  });
});
