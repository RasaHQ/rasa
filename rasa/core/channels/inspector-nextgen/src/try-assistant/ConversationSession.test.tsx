import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../tests/utils";
import { UtteranceType, type Conversation } from "../types";
import { ConversationSession } from "./ConversationSession";

// Mock child components to isolate ConversationSession
vi.mock("./ConversationLog/Message/Message", () => ({
  Message: ({ utterance, ...props }: Record<string, unknown>) => (
    <div
      data-testid="message"
      data-utterance={JSON.stringify(utterance)}
      data-props={JSON.stringify(props)}
    />
  ),
}));
vi.mock("./ConversationLog/Event", () => ({
  Event: ({ event, ...props }: Record<string, unknown>) => (
    <div
      data-testid="event"
      data-event={JSON.stringify(event)}
      data-props={JSON.stringify(props)}
    />
  ),
}));
vi.mock("./ConversationLoadingSpinner", () => ({
  ConversationLoadingSpinner: () => <div data-testid="spinner"></div>,
}));

describe("ConversationSession", () => {
  let baseConversation: Conversation;

  beforeEach(() => {
    baseConversation = {
      id: "conv1",
      startDate: new Date().toISOString(),
      events: [
        {
          id: "1",
          type: UtteranceType.User,
          text: "Hello",
        },
        {
          id: "2",
          type: UtteranceType.Bot,
          text: "Hi there!",
        },
      ],
      // ...other props as required
    } as Conversation;
  });

  it("renders session start date label", () => {
    renderWithProviders(
      <ConversationSession
        conversation={baseConversation}
        inspectorMode={false}
      />,
    );
    expect(screen.getByText(/Session started on/)).toBeInTheDocument();
  });

  it("renders Message components for user and bot utterances", () => {
    renderWithProviders(
      <ConversationSession
        conversation={baseConversation}
        inspectorMode={false}
      />,
    );
    const messages = screen.getAllByTestId("message");
    expect(messages).toHaveLength(2);
    expect(messages[0]).toHaveAttribute(
      "data-utterance",
      expect.stringContaining("Hello"),
    );
    expect(messages[1]).toHaveAttribute(
      "data-utterance",
      expect.stringContaining("Hi there!"),
    );
  });

  it("marks Message as interactive only for the last bot utterance", () => {
    renderWithProviders(
      <ConversationSession
        conversation={baseConversation}
        inspectorMode={false}
        interactive={true}
      />,
    );
    const messages = screen.getAllByTestId("message");
    expect(messages[1].dataset.props).toContain('"isInteractive":true');
    expect(messages[0].dataset.props).not.toContain('"isInteractive":true');
  });

  it("renders ConversationLoadingSpinner if waitingForResponse", () => {
    renderWithProviders(
      <ConversationSession
        conversation={baseConversation}
        inspectorMode={false}
        waitingForResponse
      />,
    );
    expect(screen.getByTestId("spinner")).toBeVisible();
  });

  it("highlights selected message/event by selectedElementId", () => {
    renderWithProviders(
      <ConversationSession
        conversation={baseConversation}
        inspectorMode={true}
        selectedElementId="2"
        selectable
      />,
    );
    const messages = screen.getAllByTestId("message");
    expect(messages[1].dataset.props).toContain('"isSelected":true');
  });
});
