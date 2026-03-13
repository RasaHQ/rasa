import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../tests/utils";
import { UtteranceType, type Conversation } from "../types";
import { TryAssistantConversation } from "./TryAssistantConversation";

vi.mock("../VerticalScroll", () => ({
  ScrollContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ScrollContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("./ConversationSession", () => ({
  ConversationSession: ({ waitingForResponse, ...props }: Record<string, unknown>) => (
    <div
      data-testid="conversation-session"
      data-waiting-for-response={String(waitingForResponse)}
      data-props={JSON.stringify(props)}
    />
  ),
}));

describe("TryAssistantConversation", () => {
  let conversationWithUserMessages: Conversation;
  let conversationWithoutUserMessages: Conversation;

  beforeEach(() => {
    conversationWithUserMessages = {
      id: "conv1",
      startDate: new Date().toISOString(),
      reviewed: false,
      totalNumberOfUserMessages: 1,
      events: [
        {
          __typename: "Utterance",
          id: "1",
          type: UtteranceType.User,
          text: "Hello",
          timestamp: new Date().toISOString(),
          tokens: [],
          rephrase: false,
          rephrasePrompt: null,
          metadata: { parseData: {} },
        },
      ],
    };

    conversationWithoutUserMessages = {
      id: "conv2",
      startDate: new Date().toISOString(),
      reviewed: false,
      totalNumberOfUserMessages: 0,
      events: [],
    };
  });

  describe("waitingForResponse behavior", () => {
    it("does not pass waitingForResponse when no user messages have been sent", () => {
      renderWithProviders(
        <TryAssistantConversation
          conversationList={[conversationWithoutUserMessages]}
          inspectorMode={false}
          waitingForResponse={true}
        />,
      );

      const session = screen.getByTestId("conversation-session");
      expect(session).toHaveAttribute("data-waiting-for-response", "false");
    });

    it("passes waitingForResponse when user has sent at least one message", () => {
      renderWithProviders(
        <TryAssistantConversation
          conversationList={[conversationWithUserMessages]}
          inspectorMode={false}
          waitingForResponse={true}
        />,
      );

      const session = screen.getByTestId("conversation-session");
      expect(session).toHaveAttribute("data-waiting-for-response", "true");
    });

    it("does not pass waitingForResponse when waitingForResponse prop is false", () => {
      renderWithProviders(
        <TryAssistantConversation
          conversationList={[conversationWithUserMessages]}
          inspectorMode={false}
          waitingForResponse={false}
        />,
      );

      const session = screen.getByTestId("conversation-session");
      expect(session).toHaveAttribute("data-waiting-for-response", "false");
    });
  });

  describe("multiple conversations", () => {
    it("only passes waitingForResponse to the last conversation", () => {
      const secondConversation: Conversation = {
        ...conversationWithUserMessages,
        id: "conv3",
      };

      renderWithProviders(
        <TryAssistantConversation
          conversationList={[conversationWithUserMessages, secondConversation]}
          inspectorMode={false}
          waitingForResponse={true}
        />,
      );

      const sessions = screen.getAllByTestId("conversation-session");
      expect(sessions).toHaveLength(2);
      expect(sessions[0]).toHaveAttribute("data-waiting-for-response", "false");
      expect(sessions[1]).toHaveAttribute("data-waiting-for-response", "true");
    });
  });
});
