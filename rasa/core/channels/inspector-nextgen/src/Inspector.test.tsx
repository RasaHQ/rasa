import { screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "./tests/utils";
import { Inspector } from "./Inspector";
import type { ConversationEventAction } from "./types";

const mockUseBotConnection = vi.fn();
vi.mock("./hooks/useBotConnection", () => ({
  useBotConnection: (...args: unknown[]) => mockUseBotConnection(...args) as Record<string, unknown>,
}));

vi.mock("./try-assistant/TryAssistant", () => ({
  TryAssistant: (props: Record<string, unknown>) => (
    <div data-testid="try-assistant" data-props={JSON.stringify(props)} />
  ),
}));

vi.mock("uuid", () => ({
  v4: () => "generated-uuid",
}));

const defaultBotConnection = {
  sendMessage: vi.fn(),
  setUrl: vi.fn(),
  startNewConversation: vi.fn(),
  conversationList: [],
  inputDisabled: false,
  sessionId: "sess-1",
  stack: [],
  replayingConversation: false,
  waitingForUserInput: true,
  replayConversation: vi.fn(),
  startVoiceStreaming: vi.fn(),
  stopVoiceStreaming: vi.fn(),
};

function getTryAssistantProps(): Record<string, unknown> {
  const el = screen.getByTestId("try-assistant");
  return JSON.parse(el.dataset.props ?? "{}") as Record<string, unknown>;
}

describe("Inspector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseBotConnection.mockReturnValue(defaultBotConnection);
  });

  it("renders TryAssistant", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(screen.getByTestId("try-assistant")).toBeInTheDocument();
  });

  it("passes projectUrl and botDataEndpoint to TryAssistant", () => {
    renderWithProviders(
      <Inspector projectUrl="http://bot.test" botDataEndpoint="/api/data" />,
    );

    const props = getTryAssistantProps();
    expect(props.projectUrl).toBe("http://bot.test");
    expect(props.botDataEndpoint).toBe("/api/data");
  });

  it("passes useBotConnection results to TryAssistant", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    const props = getTryAssistantProps();
    expect(props.sessionId).toBe("sess-1");
    expect(props.inputDisabled).toBe(false);
    expect(props.conversationList).toEqual([]);
    expect(props.replayingConversation).toBe(false);
    expect(props.waitingForUserInput).toBe(true);
  });

  it("uses generated uuid when projectId is not provided", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ projectId: "generated-uuid" }),
    );
  });

  it("uses provided projectId instead of generated uuid", () => {
    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        projectId="custom-project-id"
      />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ projectId: "custom-project-id" }),
    );
  });

  it("passes singleSessionMode as useMemoryOnly to useBotConnection", () => {
    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        singleSessionMode={true}
      />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ useMemoryOnly: true }),
    );
  });

  it("defaults useMemoryOnly to false when singleSessionMode is not set", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ useMemoryOnly: false }),
    );
  });

  it("sets flowView to false by default", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(getTryAssistantProps().flowView).toBe(false);
  });

  it("sets flowView to true when initialInspectMode is true", () => {
    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        initialInspectMode={true}
      />,
    );

    expect(getTryAssistantProps().flowView).toBe(true);
  });

  it("calls onInspectModeChange with initial inspect mode value", async () => {
    const onInspectModeChange = vi.fn();

    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        initialInspectMode={true}
        onInspectModeChange={onInspectModeChange}
      />,
    );

    await waitFor(() => {
      expect(onInspectModeChange).toHaveBeenCalledWith(true);
    });
  });

  it("defaults voiceFeaturesEnabled to true", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(getTryAssistantProps().voiceFeaturesEnabled).toBe(true);
  });

  it("passes voiceFeaturesEnabled=false to TryAssistant", () => {
    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        voiceFeaturesEnabled={false}
      />,
    );

    expect(getTryAssistantProps().voiceFeaturesEnabled).toBe(false);
  });

  it("passes conversationEventActions to TryAssistant", () => {
    const actions: ConversationEventAction[] = [
      {
        label: "test",
        icon: { prefix: "fas", iconName: "check", icon: [512, 512, [], "f00c", ""] },
        action: vi.fn(),
      },
    ];

    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        conversationEventActions={actions}
      />,
    );

    const props = getTryAssistantProps();
    expect(props.conversationEventActions).toEqual(
      expect.arrayContaining([expect.objectContaining({ label: "test" })]),
    );
  });

  it("forwards onSessionStart callback to useBotConnection", () => {
    const onSessionStart = vi.fn();

    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        onSessionStart={onSessionStart}
      />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ onSessionStart }),
    );
  });

  it("forwards onReconnectError callback to useBotConnection", () => {
    const onReconnectError = vi.fn();

    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        onReconnectError={onReconnectError}
      />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ onReconnectError }),
    );
  });

  it("forwards onMessageSent callback to useBotConnection", () => {
    const onMessageSent = vi.fn();

    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        onMessageSent={onMessageSent}
      />,
    );

    expect(mockUseBotConnection).toHaveBeenCalledWith(
      expect.objectContaining({ onMessageSent }),
    );
  });
});
