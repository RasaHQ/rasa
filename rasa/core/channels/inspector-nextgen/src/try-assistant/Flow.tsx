import { useMemo } from "react";
import { ReactFlowProvider } from "reactflow";
import { CanvasContextProvider } from "../CanvasContext";
import { Canvas } from "../canvas/Canvas";
import {
  type Conversation,
  type FlowEdge,
  type FlowNode,
  type UnionEventType,
  type Flow,
} from "../types";
import { flowStepTrail } from "../utils";

interface Props {
  flow: Flow;
  selectedNodeId?: string;
  conversation?: Conversation;
}

// Helper function to find the shortest path using breadth-first search in a graph
function findShortestPath(
  edges: FlowEdge[],
  start: string,
  end: string,
): string[] | null {
  const queue: string[][] = [[start]];
  const visited = new Set<string>();

  while (queue.length > 0) {
    const path = queue.shift()!;
    const node = path[path.length - 1];

    if (node === end) {
      return path;
    }

    if (!visited.has(node)) {
      visited.add(node);

      const neighbors = edges
        .filter((edge) => edge.sourceId === node)
        .map((edge) => edge.targetId);

      for (const neighbor of neighbors) {
        queue.push([...path, neighbor]);
      }
    }
  }

  return null; // No path found
}

function findSourceNode(
  nodes: FlowNode[],
  edge: FlowEdge,
): FlowNode | undefined {
  return nodes.find((node) => node.id === edge.sourceId);
}

function findTargetNode(
  nodes: FlowNode[],
  edge: FlowEdge,
): FlowNode | undefined {
  return nodes.find((node) => node.id === edge.targetId);
}

function collectHighlightedEdges(
  events: UnionEventType[],
  flow: Flow,
): { highlightedEdges: FlowEdge[]; highlightedNodes: FlowNode[] } {
  const proFlowStepTrail: string[] = flowStepTrail(events)[flow.id || ""] || [];

  // Use a Set to track unique edge combinations to avoid duplicates
  const highlightedEdgesCombinationSet = new Set<string>();
  const highlightedEdges: FlowEdge[] = [];
  const highlightedNodes: FlowNode[] = [];
  let previousHighlightedStep: string | null =
    proFlowStepTrail.length > 0 ? proFlowStepTrail[0] : null;

  for (let i = 0; i < proFlowStepTrail.length - 1; i++) {
    const currentStep = proFlowStepTrail[i];
    const nextStep = proFlowStepTrail[i + 1];

    // Create a unique key for this edge to avoid duplicates
    const edgeKey = `${currentStep}->${nextStep}`;
    if (highlightedEdgesCombinationSet.has(edgeKey)) {
      continue;
    }

    // Check if there is a direct edge
    const directEdge = flow.edges?.find(
      (edge) => edge.sourceId === currentStep && edge.targetId === nextStep,
    );

    if (directEdge) {
      highlightedEdgesCombinationSet.add(edgeKey);
      highlightedEdges.push(directEdge);
      const sourceNode = findSourceNode(flow.nodes || [], directEdge);
      if (sourceNode) {
        highlightedNodes.push(sourceNode);
      }
      const targetNode = findTargetNode(flow.nodes || [], directEdge);
      if (targetNode) {
        highlightedNodes.push(targetNode);
      }
      previousHighlightedStep = nextStep;
      continue;
    }

    // Find the shortest path if no direct edge
    // this is necessary since studio has additional nodes like "if" that
    // PRO doesn't have, so we need to find the shortest path between the
    // two steps PRO reports
    // this is a heuristic approach and might not always be correct!
    let path = findShortestPath(flow.edges || [], currentStep, nextStep);

    if (!path) {
      if (!previousHighlightedStep) {
        continue;
      }
      // sometimes, there are paths where PRO has a node that is not
      // present in studio (noop steps), so we try to find a path
      // that goes through the previous highlighted step and essentially skips
      // the current step
      path = findShortestPath(
        flow.edges || [],
        previousHighlightedStep,
        nextStep,
      );
      if (!path) {
        // no path found, so we skip this step
        continue;
      }
    }

    highlightedEdgesCombinationSet.add(edgeKey);

    for (let j = 0; j < path.length - 1; j++) {
      const edge = flow.edges?.find(
        (e) => e.sourceId === path[j] && e.targetId === path[j + 1],
      );
      if (edge) {
        const pathEdgeKey = `${edge.sourceId}->${edge.targetId}`;
        if (!highlightedEdgesCombinationSet.has(pathEdgeKey)) {
          highlightedEdgesCombinationSet.add(pathEdgeKey);
          highlightedEdges.push(edge);
          const sourceNode = findSourceNode(flow.nodes || [], edge);
          if (sourceNode) {
            highlightedNodes.push(sourceNode);
          }
          const targetNode = findTargetNode(flow.nodes || [], edge);
          if (targetNode) {
            highlightedNodes.push(targetNode);
          }
        }
        previousHighlightedStep = path[j + 1];
      }
    }
  }

  return {
    highlightedEdges: highlightedEdges,
    highlightedNodes: highlightedNodes,
  };
}

export function Flow({ flow, conversation }: Props) {
  const { highlightedEdges, highlightedNodes } = useMemo(() => {
    if (!flow) {
      return { highlightedEdges: [], highlightedNodes: [] };
    }
    return collectHighlightedEdges(conversation?.events ?? [], flow);
  }, [conversation?.events, flow]);

  return (
    <ReactFlowProvider>
      <CanvasContextProvider
        flowName={flow.name || flow.id}
        initNodes={flow?.nodes}
        initEdges={flow?.edges}
        highlightedEdges={highlightedEdges}
        highlightedNodes={highlightedNodes}
      >
        <Canvas />
      </CanvasContextProvider>
    </ReactFlowProvider>
  );
}
