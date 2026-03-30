import { describe, it, vi, expect, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../tests/utils";
import { TryAssistant } from "./TryAssistant";

vi.mock("../hooks/useConversationData", () => ({
  useConversationData: vi.fn(),
}));

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
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders chat section and doesn't render spinner or flow section by default", () => {
    renderWithProviders(<TryAssistant />, {
      initialStoreState: { inspectMode: false },
    });
    expect(screen.getByTestId("chat-section")).toBeInTheDocument();
    expect(screen.queryByTestId("spinner")).not.toBeInTheDocument();
    expect(screen.queryByTestId("flow-section")).not.toBeInTheDocument();
  });

  it("renders flow section if inspectMode is true in the store", () => {
    renderWithProviders(<TryAssistant />, {
      initialStoreState: { inspectMode: true },
    });
    expect(screen.getByTestId("flow-section")).toBeInTheDocument();
  });

  it("calls onInspectModeChange when inspectMode is set in store", async () => {
    const onInspectModeChange = vi.fn();
    renderWithProviders(
      <TryAssistant onInspectModeChange={onInspectModeChange} />,
      { initialStoreState: { inspectMode: true } },
    );
    await waitFor(() => {
      expect(onInspectModeChange).toHaveBeenCalledWith(true);
    });
  });

  it("when selectedElement is undefined, shows latestStack", () => {
    const latestStack = {
      frameId: "from-stack",
      flowId: "f",
      stepId: "s",
      ended: true,
    };
    renderWithProviders(<TryAssistant />, {
      initialStoreState: {
        inspectMode: true,
        stack: [latestStack],
        conversationList: [],
      },
    });
    const flow = screen.getByTestId("flow-section");
    const props = JSON.parse(flow.dataset.props ?? "{}") as {
      stackToShow: typeof latestStack;
    };
    expect(props.stackToShow).toEqual(latestStack);
  });
});
