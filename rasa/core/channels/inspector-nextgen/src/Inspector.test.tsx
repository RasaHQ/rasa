import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "./tests/utils";
import { Inspector } from "./Inspector";

const mockUseBotConnection = vi.fn();
vi.mock("./hooks/useBotConnection", () => ({
  useBotConnection: (...args: unknown[]) =>
    mockUseBotConnection(...args) as void,
}));

const mockUseTrackerConnection = vi.fn();
vi.mock("./hooks/useTrackerConnection", () => ({
  useTrackerConnection: (...args: unknown[]) =>
    mockUseTrackerConnection(...args) as void,
}));

vi.mock("./hooks/useConversationData", () => ({
  useConversationData: vi.fn(),
}));

vi.mock("./try-assistant/TryAssistant", () => ({
  TryAssistant: (props: Record<string, unknown>) => (
    <div data-testid="try-assistant" data-props={JSON.stringify(props)} />
  ),
}));

vi.mock("uuid", () => ({
  v4: () => "generated-uuid",
}));

describe("Inspector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders TryAssistant", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(screen.getByTestId("try-assistant")).toBeInTheDocument();
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

  it("initializes store with inspectMode=true when initialInspectMode is true", () => {
    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        initialInspectMode={true}
      />,
    );

    expect(screen.getByTestId("try-assistant")).toBeInTheDocument();
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

  it("initializes store with isEmbedded=true when isEmbedded prop is true", () => {
    renderWithProviders(
      <Inspector
        projectUrl="http://localhost:5005"
        botDataEndpoint="/data"
        isEmbedded={true}
      />,
    );

    expect(screen.getByTestId("try-assistant")).toBeInTheDocument();
  });

  it("defaults isEmbedded to false when not provided", () => {
    renderWithProviders(
      <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
    );

    expect(screen.getByTestId("try-assistant")).toBeInTheDocument();
  });

  describe("channel routing", () => {
    it("enables useBotConnection and disables useTrackerConnection when no channel is provided", () => {
      renderWithProviders(
        <Inspector projectUrl="http://localhost:5005" botDataEndpoint="/data" />,
      );

      expect(mockUseBotConnection).toHaveBeenCalledWith(
        expect.objectContaining({ enabled: true }),
      );
      expect(mockUseTrackerConnection).toHaveBeenCalledWith(
        expect.objectContaining({ enabled: false, channel: "" }),
      );
    });

    it("enables useTrackerConnection and disables useBotConnection when a non-inspector channel is provided", () => {
      renderWithProviders(
        <Inspector
          projectUrl="http://localhost:5005"
          botDataEndpoint="/data"
          channel="rest"
        />,
      );

      expect(mockUseTrackerConnection).toHaveBeenCalledWith(
        expect.objectContaining({ enabled: true, channel: "rest" }),
      );
      expect(mockUseBotConnection).toHaveBeenCalledWith(
        expect.objectContaining({ enabled: false }),
      );
    });

    it("treats channel=inspector the same as no channel (uses useBotConnection)", () => {
      renderWithProviders(
        <Inspector
          projectUrl="http://localhost:5005"
          botDataEndpoint="/data"
          channel="inspector"
        />,
      );

      expect(mockUseBotConnection).toHaveBeenCalledWith(
        expect.objectContaining({ enabled: true }),
      );
      expect(mockUseTrackerConnection).toHaveBeenCalledWith(
        expect.objectContaining({ enabled: false, channel: "inspector" }),
      );
    });
  });

});
