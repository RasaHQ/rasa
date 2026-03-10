import ELK, {
  type ElkExtendedEdge,
  type ElkNode,
} from "elkjs/lib/elk.bundled.js";
import { type NodeProps, type Node } from "reactflow";
import {
  type FlowNode,
  type FlowEdge,
  FlowEdgeType,
  FlowNodeType,
  NodeConditionSubType,
} from "../types";
import { DEFAULT_NODE_WIDTH, DEFAULT_NODE_HEIGHT } from "../constants";

const elk = new ELK({
  defaultLayoutOptions: {
    "elk.algorithm": "org.eclipse.elk.layered",
    "elk.direction": "DOWN",
    "elk.layered.spacing.edgeNodeBetweenLayers": "50",
    "org.eclipse.elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
    "org.eclipse.elk.alignment": "TOP",
    "spacing.nodeNodeBetweenLayers": "40",
    "org.eclipse.elk.spacing.nodeNode": "50",
    "org.eclipse.elk.layered.considerModelOrder.strategy": "PREFER_EDGES",
  },
});

export const nodePropsToNodeObject = (nodeProps: NodeProps<FlowNode>): Node => {
  const { data, id, type, xPos, yPos, selected } = nodeProps;

  return {
    data,
    id,
    type,
    position: {
      x: xPos,
      y: yPos,
    },
    selected,
  };
};

export const extractNodeLabel = (node: FlowNode) => {
  return node.label;
};

function getPosition(elkNode: ElkNode) {
  return {
    x: Math.floor((elkNode.x || 1) - (elkNode.width || 1) / 2),
    y: Math.floor((elkNode.y || 1) - (elkNode.height || 1) / 2),
  };
}

export async function autoLayout(
  nodes: FlowNode[],
  allEdges: FlowEdge[],
): Promise<FlowNode[]> {
  const elkNodes: ElkNode[] = nodes.map((flowNode) => ({
    id: flowNode.id,
    width: DEFAULT_NODE_WIDTH,
    height: DEFAULT_NODE_HEIGHT,
  }));

  const objectNodes: { [K in FlowNode["id"]]: FlowNode } = nodes.reduce(
    (obj, item) => Object.assign(obj, { [item.id]: item }),
    {},
  );

  const meaningfulEdges = allEdges.filter(
    (edge) => edge.type === FlowEdgeType.Auto,
  );

  const orderedEdges = meaningfulEdges.sort((edgeA, edgeB) => {
    const nodeA = objectNodes[edgeA.targetId];
    const nodeB = objectNodes[edgeB.targetId];

    if (
      nodeA.type === FlowNodeType.Condition &&
      nodeA.subType === NodeConditionSubType.ELSE
    ) {
      return 1;
    }

    if (
      nodeB.type === FlowNodeType.Condition &&
      nodeB.subType === NodeConditionSubType.ELSE
    ) {
      return -1;
    }

    if (
      nodeA.type === FlowNodeType.Condition &&
      nodeB.type === FlowNodeType.Condition
    ) {
      return nodeA.order - nodeB.order;
    }

    return 1;
  });

  const elkEdges: ElkExtendedEdge[] = orderedEdges.map((flowEdge) => ({
    id: `${flowEdge.targetId}-${flowEdge.sourceId}`,
    targets: [flowEdge.targetId],
    sources: [flowEdge.sourceId],
  }));

  const rootNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "org.eclipse.elk.layered",
      "elk.direction": "DOWN",
    },
    children: elkNodes,
    edges: elkEdges,
  };

  const newGraph = await elk.layout(rootNode);

  if (!newGraph?.children) {
    throw new Error("Invalid graph");
  }

  const graphChildrenDictionary = Object.fromEntries(
    newGraph.children.map((child) => [child.id, child]),
  );

  return nodes.map((node) => {
    const elkNode = graphChildrenDictionary[node.id];
    const { x, y } = getPosition(elkNode);
    return {
      ...node,
      metadata: {
        x,
        y,
      },
    };
  });
}

