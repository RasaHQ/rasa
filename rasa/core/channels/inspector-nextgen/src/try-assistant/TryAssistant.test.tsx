import { describe, it, vi, expect, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../tests/utils";
import { TryAssistant } from "./TryAssistant";
import { InspectorView } from "../types/inspector";

const mockIsLargeScreen = vi.fn(() => false);
vi.mock("../hooks/useIsLargeScreen", () => ({
  useIsLargeScreen: () => mockIsLargeScreen(),
}));

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
vi.mock("./HistorySection", () => ({
  HistorySection: (props: Record<string, unknown>) => (
    <div data-testid="history-section" data-props={JSON.stringify(props)} />
  ),
}));
vi.mock("./MemorySection", () => ({
  MemorySection: (props: Record<string, unknown>) => (
    <div data-testid="memory-section" data-props={JSON.stringify(props)} />
  ),
}));
vi.mock("./EventDetails", () => ({
  EventDetails: () => <div data-testid="event-details" />,
}));

describe("TryAssistant", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsLargeScreen.mockReturnValue(false);
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
      initialStoreState: {
        inspectMode: true,
        inspectorView: InspectorView.ActiveFlow,
      },
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
        inspectorView: InspectorView.ActiveFlow,
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

  describe("small screen (default) layout", () => {
    it("renders HistorySection when inspectorView is History", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.History,
        },
      });
      expect(screen.getByTestId("history-section")).toBeInTheDocument();
      expect(screen.queryByTestId("flow-section")).not.toBeInTheDocument();
    });

    it("renders MemorySection when inspectorView is Memory", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.Memory,
        },
      });
      expect(screen.getByTestId("memory-section")).toBeInTheDocument();
      expect(screen.queryByTestId("flow-section")).not.toBeInTheDocument();
    });

    it("does not show inspector-side-panel on small screen", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.ActiveFlow,
        },
      });
      expect(
        screen.queryByTestId("inspector-side-panel"),
      ).not.toBeInTheDocument();
    });
  });

  describe("All layout (large screen)", () => {
    beforeEach(() => {
      mockIsLargeScreen.mockReturnValue(true);
    });

    it("renders 3-column layout when inspectMode + large screen + All view", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.All,
        },
      });
      expect(screen.getByTestId("chat-section")).toBeInTheDocument();
      expect(screen.getByTestId("inspector-canvas")).toBeInTheDocument();
      expect(screen.getByTestId("inspector-side-panel")).toBeInTheDocument();
    });

    it("shows flow section, history, and memory simultaneously", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.All,
        },
      });
      expect(screen.getByTestId("flow-section")).toBeInTheDocument();
      expect(screen.getByTestId("history-section")).toBeInTheDocument();
      expect(screen.getByTestId("memory-section")).toBeInTheDocument();
    });

    it("passes showViewSwitcher=false to MemorySection in All layout", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.All,
        },
      });
      const mem = screen.getByTestId("memory-section");
      const props = JSON.parse(mem.dataset.props ?? "{}") as {
        showViewSwitcher: boolean;
      };
      expect(props.showViewSwitcher).toBe(false);
    });

    it("shows EventDetails instead of history/memory when element is selected", () => {
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.All,
          selectedElement: { id: "ev-1", event: "action" } as never,
        },
      });
      expect(screen.getByTestId("event-details")).toBeInTheDocument();
      expect(
        screen.queryByTestId("history-section"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("memory-section"),
      ).not.toBeInTheDocument();
    });
  });

  describe("auto-view switching effects", () => {
    it("falls back to ActiveFlow view when screen becomes small while on All", async () => {
      mockIsLargeScreen.mockReturnValue(false);
      renderWithProviders(<TryAssistant />, {
        initialStoreState: {
          inspectMode: true,
          inspectorView: InspectorView.All,
        },
      });

      await waitFor(() => {
        expect(screen.queryByTestId("inspector-side-panel")).not.toBeInTheDocument();
      });
    });
  });
});
