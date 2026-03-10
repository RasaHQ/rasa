import { describe, it, vi, expect, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../tests/utils";
import { TryAssistant } from "./TryAssistant";

vi.mock("@tanstack/react-store", async () => {
  const actual = await vi.importActual("@tanstack/react-store");
  return { ...actual, useStore: vi.fn(() => "mock-url") };
});

// Mock children components
vi.mock("../LoadingSpinner", () => ({
  LoadingSpinner: () => <div data-testid="spinner" />,
}));
vi.mock("./ChatSection", () => ({
  ChatSection: (props: Record<string, unknown>) => (
    <div data-testid="chat-section" data-props={JSON.stringify(props)} />
  ),
}));
vi.mock("./FlowSection", () => ({
  FlowSection: (props: Record<string, unknown>) => {
    return (
      <div data-testid="flow-section" data-props={JSON.stringify(props)} />
    );
  },
}));

describe("TryAssistant", () => {
  const baseProps = {
    flowView: false,
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
    projectUrl: "mock-url",
    conversationEventActions: [],
    setFlowView: vi.fn(),
    botDataEndpoint: "/data",
    startVoiceStreaming: vi.fn(),
    stopVoiceStreaming: vi.fn(),
    voiceFeaturesEnabled: true,
  } satisfies Parameters<typeof TryAssistant>[0];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders chat section and doesn't render spinner or flow section by default", () => {
    renderWithProviders(<TryAssistant {...baseProps} />);
    expect(screen.getByTestId("chat-section")).toBeInTheDocument();
    expect(screen.queryByTestId("spinner")).not.toBeInTheDocument();
    expect(screen.queryByTestId("flow-section")).not.toBeInTheDocument();
  });

  it("renders flow section if flowView is true", () => {
    renderWithProviders(<TryAssistant {...baseProps} flowView={true} />);
    expect(screen.getByTestId("flow-section")).toBeInTheDocument();
  });

  it("calls setUrl with projectUrl in useEffect", () => {
    renderWithProviders(
      <TryAssistant {...baseProps} setUrl={baseProps.setUrl} />,
    );
    expect(baseProps.setUrl).toHaveBeenCalledWith("mock-url");
  });

  it("when selectedElement is undefined, shows latestStack", () => {
    const latestStack = {
      frameId: "from-stack",
      flowId: "f",
      stepId: "s",
      ended: true,
    };
    renderWithProviders(
      <TryAssistant
        {...baseProps}
        flowView
        stack={[latestStack]}
        conversationList={[]}
      />,
    );
    const flow = screen.getByTestId("flow-section");
    const props = JSON.parse(flow.dataset.props ?? "{}") as {
      stackToShow: typeof latestStack;
    };
    expect(props.stackToShow).toEqual(latestStack);
  });
});
