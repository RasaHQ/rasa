import { debounce } from "lodash";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  type Edge,
  type Node,
  type ReactFlowInstance,
  MarkerType,
  useEdgesState,
  useNodesState,
  useReactFlow,
} from "reactflow";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../constants";
import { useTheme } from "../hooks/useTheme";
import {
  type FlowEdge,
  type FlowNode,
  FlowEdgeType,
  FlowNodeType,
} from "../types";
import { autoLayout } from "../utils";
import { useInspectorContext } from "../InspectorContext";
import { Context } from "./CanvasContext";

interface Props {
  flowName: string;
  initNodes: FlowNode[];
  initEdges: FlowEdge[];
  highlightedEdges?: FlowEdge[];
  highlightedNodes?: FlowNode[];
  children: ReactNode;
}

export const CanvasContextProvider = ({
  children,
  flowName,
  initNodes = [],
  initEdges = [],
  highlightedEdges = [],
  highlightedNodes = [],
}: Props) => {
  const { logError } = useInspectorContext();
  const { getToken } = useTheme();
  const markerEndColor = getToken("colors.rasaNeutral.500") as string;
  const highlightedMarkerEndColor = getToken(
    "colors.rasawebDeepPurple.800",
  ) as string;
  const { zoomIn, zoomOut, fitView, zoomTo } = useReactFlow();
  const [nodesWithCoordinates, setNodesWithCoordinates] = useState(
    [] as FlowNode[],
  );
  const [flowInstance, setFlowInstance] = useState<ReactFlowInstance | null>(
    null,
  );
  const [nodes, setNodes, handleNodesChange] = useNodesState(
    nodesApiToCanvas(nodesWithCoordinates, initEdges, highlightedNodes),
  );
  const [edges, setEdges] = useEdgesState(
    edgesApiToCanvas(
      initEdges,
      highlightedEdges,
      markerEndColor,
      highlightedMarkerEndColor,
    ),
  );

  const selectedNode = useMemo(() => findSelectedNode(nodes), [nodes]);

  useEffect(() => {
    setEdges(
      edgesApiToCanvas(
        initEdges,
        highlightedEdges,
        markerEndColor,
        highlightedMarkerEndColor,
      ),
    );
  }, [
    initEdges,
    highlightedEdges,
    setEdges,
    markerEndColor,
    highlightedMarkerEndColor,
  ]);

  const initializeNodes = useCallback(
    (initNodes: FlowNode[], initEdges: FlowEdge[]) => {
      autoLayout(initNodes, initEdges)
        .then((nodesWithCoords) => setNodesWithCoordinates(nodesWithCoords))
        .catch((error) => {
          logError(error, {
            tags: {
              component: "CanvasContextProvider",
              action: "initializeNodes",
            },
          });
        });
    },
    [setNodesWithCoordinates, logError],
  );

  useEffect(() => {
    initializeNodes(initNodes, initEdges);
  }, [initNodes, initEdges, initializeNodes]);

  useEffect(() => {
    if (nodesWithCoordinates.length > 0) {
      const newNodes = nodesApiToCanvas(
        nodesWithCoordinates,
        initEdges,
        highlightedNodes,
      );
      setNodes(newNodes);
    }
  }, [nodesWithCoordinates, initEdges, setNodes, highlightedNodes]);

  const focusOnNode = useCallback(
    (node: Node | undefined | null = null, animate?: boolean) => {
      if (!flowInstance || !node) {
        return;
      }

      const { zoom } = flowInstance.getViewport();
      const duration = animate ? 200 : 0;
      const x = node.position.x + DEFAULT_NODE_WIDTH / 2;
      const y = node.position.y + DEFAULT_NODE_HEIGHT / 2;

      flowInstance.setCenter(x, y, { zoom, duration });
    },
    [flowInstance],
  );

  useEffect(() => {
    const debouncedResize = debounce(function resize() {
      focusOnNode(selectedNode ?? nodes[0], true);
    }, 500);

    globalThis.addEventListener("resize", debouncedResize);

    return () => {
      globalThis.removeEventListener("resize", debouncedResize);
    };
  }, [selectedNode, nodes, focusOnNode]);

  const timeoutRef = useRef<ReturnType<typeof globalThis.setTimeout>>(null);

  useEffect(() => {
    timeoutRef.current = setTimeout(
      () => (selectedNode ? focusOnNode(selectedNode, true) : null),
      0,
    );

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [focusOnNode, selectedNode]);

  const initialFitDoneRef = useRef(false);

  useEffect(() => {
    initialFitDoneRef.current = false;
  }, [initNodes]);

  useEffect(() => {
    if (flowInstance && nodesWithCoordinates.length > 0 && !initialFitDoneRef.current) {
      initialFitDoneRef.current = true;
      setTimeout(() => {
        fitView({ minZoom: 0.1 });
      }, 0);
    }
  }, [flowInstance, nodesWithCoordinates, fitView]);

  const handleZoomInClick = () => {
    zoomIn({ duration: 200 });
  };

  const handleZoomOutClick = () => {
    zoomOut({ duration: 200 });
  };

  const handleZoomTo100 = () => {
    zoomTo(1, { duration: 200 });
  };

  const handleFitToCanvasClick = () => {
    fitView({ minZoom: 0.1 });
  };

  const handleInit = useCallback(
    (instance: ReactFlowInstance) => {
      setFlowInstance(instance);
      focusOnNode(instance.getNodes()?.[0], false);
    },
    [setFlowInstance, focusOnNode],
  );

  const value = {
    flowName,
    nodes,
    edges,
    handleInit,
    focusOnNode,
    handleNodesChange,
    handleZoomInClick,
    handleZoomOutClick,
    handleZoomTo100,
    handleFitToCanvasClick,
  };

  return <Context.Provider value={value}> {children} </Context.Provider>;
};

function findSelectedNode(nodes: Node[]) {
  const selectedNode = nodes.findLast((node) => node.selected);
  return selectedNode ?? null;
}

function nodesApiToCanvas(
  nodes: FlowNode[],
  edges: FlowEdge[],
  highlightedNodes: FlowNode[],
  isReadOnly: boolean = true,
): Node<FlowNode>[] {
  const startNodeType = nodes.length === 1 ? "startAlone" : "start";

  return nodes.map((node) => {
    const outboundEdges = edges.filter(({ sourceId }) => sourceId === node.id);
    const isLeafNode = outboundEdges.length === 0;
    const isStartNode = node.type === FlowNodeType.Start;
    const isHighlighted = highlightedNodes?.some(
      (highlightedNode) => highlightedNode.id === node.id,
    );
    let data: FlowNode;

    switch (node.type) {
      case FlowNodeType.Start:
        data = { ...node };
        break;
      case FlowNodeType.CollectInformation:
        data = { ...node };
        break;
      case FlowNodeType.CustomAction:
        data = { ...node };
        break;
      case FlowNodeType.Logic:
        data = { ...node };
        break;
      case FlowNodeType.Link:
        data = { ...node };
        break;
      case FlowNodeType.Message:
        data = { ...node };
        break;
      case FlowNodeType.Condition:
        data = { ...node };
        break;
      case FlowNodeType.SetSlots:
        data = { ...node };
        break;
      case FlowNodeType.Call:
        data = { ...node };
        break;
      default:
        data = { ...(node as FlowNode) };
    }

    return {
      id: node.id,
      position: {
        x: node.metadata.x,
        y: node.metadata.y,
      },
      type: isStartNode ? startNodeType : isLeafNode ? "end" : "standard",
      data: {
        ...data,
        isReadOnly,
      },
      selected: !!isHighlighted,
      selectable: false,
    };
  });
}

function edgesApiToCanvas(
  edges: FlowEdge[],
  highlightedEdges: FlowEdge[],
  markerEndColor: string,
  highlightedMarkerEndColor: string,
): Edge[] {
  return edges.map((edge) => {
    let type = "add";

    if (edge.type === FlowEdgeType.Custom) {
      type = "custom";
    }
    const isHighlighted = highlightedEdges?.includes(edge);

    let edgeMarkerEndColor;
    if (isHighlighted) {
      edgeMarkerEndColor = highlightedMarkerEndColor;
    } else {
      edgeMarkerEndColor = markerEndColor;
    }

    return {
      id: `${edge.sourceId}-${edge.targetId}`,
      source: edge.sourceId,
      target: edge.targetId,
      type,
      selected: isHighlighted ?? false,
      markerEnd: {
        type: MarkerType.Arrow,
        width: 20,
        height: 20,
        color: edgeMarkerEndColor,
      },
      data: {
        isReadOnly: true,
      },
      pathOptions: { borderRadius: 8 },
    };
  });
}
