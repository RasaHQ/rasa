import { render, screen, act } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { type FlowEdge, type FlowNode, FlowEdgeType, FlowNodeType } from "../types";
import { CanvasContextProvider } from "./CanvasContextProvider";
import { useCanvasContext } from "./useCanvasContext";

// --- mocks ---

const mockZoomIn = vi.fn();
const mockZoomOut = vi.fn();
const mockFitView = vi.fn();
const mockZoomTo = vi.fn();
const mockSetNodes = vi.fn();
const mockSetEdges = vi.fn();

vi.mock("reactflow", () => ({
  useReactFlow: () => ({
    zoomIn: mockZoomIn,
    zoomOut: mockZoomOut,
    fitView: mockFitView,
    zoomTo: mockZoomTo,
  }),
  useNodesState: (initial: unknown[]) => [initial, mockSetNodes, vi.fn()],
  useEdgesState: (initial: unknown[]) => [initial, mockSetEdges],
  MarkerType: { Arrow: "arrow" },
}));

const mockAutoLayout = vi.fn();
vi.mock("../utils", () => ({
  autoLayout: (...args: unknown[]) =>
    mockAutoLayout(...args) as Promise<FlowNode[]>,
}));

const mockLogError = vi.fn();
vi.mock("../InspectorContext", () => ({
  useInspectorContext: () => ({ logError: mockLogError }),
}));

vi.mock("../hooks/useTheme", () => ({
  useTheme: () => ({
    getToken: vi.fn((token: string) =>
      token === "radii.lg" ? "8px" : "#aabbcc",
    ),
    getTokenPx: vi.fn().mockReturnValue(20),
  }),
}));

// --- helpers ---

type CanvasEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  selected: boolean;
};

const makeStartNode = (id: string): FlowNode =>
  ({ id, label: id, type: FlowNodeType.Start, metadata: { x: 0, y: 0 } }) as FlowNode;

const makeMessageNode = (id: string): FlowNode =>
  ({ id, label: id, type: FlowNodeType.Message, metadata: { x: 0, y: 0 } }) as FlowNode;

const makeEdge = (
  sourceId: string,
  targetId: string,
  type = FlowEdgeType.Auto,
): FlowEdge => ({ sourceId, targetId, type });

function ContextConsumer() {
  const ctx = useCanvasContext();
  return (
    <div
      data-testid="ctx"
      data-flow-name={ctx.flowName}
      data-edges={JSON.stringify(ctx.edges)}
      data-nodes={JSON.stringify(ctx.nodes)}
    >
      <button onClick={ctx.handleZoomInClick}>zoom-in</button>
      <button onClick={ctx.handleZoomOutClick}>zoom-out</button>
      <button onClick={ctx.handleZoomTo100}>zoom-100</button>
      <button onClick={ctx.handleFitToCanvasClick}>fit</button>
    </div>
  );
}

function renderProvider(props: {
  initNodes?: FlowNode[];
  initEdges?: FlowEdge[];
  highlightedEdges?: FlowEdge[];
  highlightedNodes?: FlowNode[];
  flowName?: string;
}) {
  return render(
    <CanvasContextProvider
      flowName={props.flowName ?? "test-flow"}
      initNodes={props.initNodes ?? []}
      initEdges={props.initEdges ?? []}
      highlightedEdges={props.highlightedEdges}
      highlightedNodes={props.highlightedNodes}
    >
      <ContextConsumer />
    </CanvasContextProvider>,
  );
}

// --- tests ---

describe("CanvasContextProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAutoLayout.mockResolvedValue([]);
  });

  it("renders children", () => {
    renderProvider({});
    expect(screen.getByTestId("ctx")).toBeInTheDocument();
  });

  it("provides flowName to context consumers", () => {
    renderProvider({ flowName: "my-flow" });
    expect(screen.getByTestId("ctx")).toHaveAttribute(
      "data-flow-name",
      "my-flow",
    );
  });

  describe("autoLayout", () => {
    it("calls autoLayout on mount with initNodes and initEdges", () => {
      const nodes = [makeStartNode("n1"), makeMessageNode("n2")];
      const edges = [makeEdge("n1", "n2")];
      renderProvider({ initNodes: nodes, initEdges: edges });
      expect(mockAutoLayout).toHaveBeenCalledWith(nodes, edges);
    });

    it("calls autoLayout again when initNodes change", () => {
      const nodes = [makeStartNode("n1")];
      const edges = [makeEdge("n1", "n2")];
      const { rerender } = renderProvider({ initNodes: nodes, initEdges: edges });
      const updatedNodes = [...nodes, makeMessageNode("n2")];
      rerender(
        <CanvasContextProvider
          flowName="test-flow"
          initNodes={updatedNodes}
          initEdges={edges}
        >
          <ContextConsumer />
        </CanvasContextProvider>,
      );
      expect(mockAutoLayout).toHaveBeenCalledTimes(2);
      expect(mockAutoLayout).toHaveBeenLastCalledWith(updatedNodes, edges);
    });

    it("logs error if autoLayout rejects", async () => {
      const error = new Error("layout failed");
      mockAutoLayout.mockRejectedValueOnce(error);
      renderProvider({ initNodes: [makeStartNode("n1")], initEdges: [] });
      await act(async () => {
        await Promise.resolve();
      });
      expect(mockLogError).toHaveBeenCalledTimes(1);
      const [[loggedErr, logContext]] = mockLogError.mock.calls as [[
        Error,
        { tags: { component: string } },
      ]];
      expect(loggedErr).toBe(error);
      expect(logContext.tags.component).toBe("CanvasContextProvider");
    });
  });

  describe("edges in context", () => {
    const getEdges = (): CanvasEdge[] =>
      JSON.parse(screen.getByTestId("ctx").getAttribute("data-edges")!) as CanvasEdge[];

    it("exposes edges with source and target derived from sourceId and targetId", () => {
      renderProvider({
        initEdges: [makeEdge("node-a", "node-b")],
      });
      const edges = getEdges();
      expect(edges).toHaveLength(1);
      expect(edges[0]).toMatchObject({ source: "node-a", target: "node-b" });
    });

    it("gives FlowEdgeType.Custom edges the canvas type 'custom'", () => {
      renderProvider({
        initEdges: [makeEdge("node-a", "node-b", FlowEdgeType.Custom)],
      });
      expect(getEdges()[0].type).toBe("custom");
    });

    it("gives FlowEdgeType.Auto edges a non-custom canvas type", () => {
      renderProvider({
        initEdges: [makeEdge("node-a", "node-b", FlowEdgeType.Auto)],
      });
      expect(getEdges()[0].type).not.toBe("custom");
    });

    it("marks highlighted edges as selected", () => {
      const edge = makeEdge("node-a", "node-b");
      renderProvider({
        initEdges: [edge],
        highlightedEdges: [edge],
      });
      expect(getEdges()[0].selected).toBe(true);
    });

    it("leaves non-highlighted edges unselected", () => {
      const edge = makeEdge("node-a", "node-b");
      const otherEdge = makeEdge("node-b", "node-c");
      renderProvider({
        initEdges: [edge, otherEdge],
        highlightedEdges: [edge],
      });
      expect(getEdges()[1].selected).toBe(false);
    });
  });

  describe("zoom handlers", () => {
    it("handleZoomInClick calls zoomIn", async () => {
      renderProvider({});
      await userEvent.click(screen.getByText("zoom-in"));
      expect(mockZoomIn).toHaveBeenCalledWith({ duration: 200 });
    });

    it("handleZoomOutClick calls zoomOut", async () => {
      renderProvider({});
      await userEvent.click(screen.getByText("zoom-out"));
      expect(mockZoomOut).toHaveBeenCalledWith({ duration: 200 });
    });

    it("handleZoomTo100 calls zoomTo with 1", async () => {
      renderProvider({});
      await userEvent.click(screen.getByText("zoom-100"));
      expect(mockZoomTo).toHaveBeenCalledWith(1, { duration: 200 });
    });

    it("handleFitToCanvasClick calls fitView", async () => {
      renderProvider({});
      await userEvent.click(screen.getByText("fit"));
      expect(mockFitView).toHaveBeenCalledWith({ minZoom: 0.1 });
    });
  });
});
