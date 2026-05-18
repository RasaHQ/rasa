import {
  type EdgeProps,
  BaseEdge,
  getSmoothStepPath,
  Position,
} from "reactflow";
import { useCanvasContext } from "../../CanvasContext";
import { FlowNodeType, type FlowNode } from "../../types";
import { useTheme } from "../../hooks/useTheme";

/**
 *
 * This function is needed because the width that is returned in the node is always the max width.
 * Maybe we can revisit this after updating reactflow to v12
 * currently the width is 11.5rem which is 184px, this is referenced in the Canvas.tsx and Content.tsx
 */
function getNodeWidth(nodeType: FlowNodeType): number {
  const maxWidth = 184;
  if (nodeType === FlowNodeType.Logic) {
    return 76.586;
  }
  return maxWidth;
}

function getEdgeCoordinatesAndPosition({
  sourceNode,
  targetNode,
  sourceX,
  sourceY,
  targetX,
  targetY,
}: {
  sourceNode: FlowNode;
  targetNode: FlowNode;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
}): {
  calculatedSourcePosition: Position;
  calculatedTargetPosition: Position;
  calculatedSourceX: number;
  calculatedSourceY: number;
  calculatedTargetX: number;
  calculatedTargetY: number;
} {
  let calculatedSourcePosition = Position.Bottom;
  let calculatedTargetPosition = Position.Bottom;
  let calculatedSourceX = sourceX;
  let calculatedSourceY = sourceY;
  let calculatedTargetX = targetX;
  let calculatedTargetY = targetY;

  const nodeHeight = 56;
  const sourceNodeWidth = getNodeWidth(sourceNode.type);
  const targetNodeWidth = getNodeWidth(targetNode.type);

  if (sourceY > targetY && sourceX > targetX) {
    calculatedSourcePosition = Position.Right;
    calculatedTargetPosition = Position.Right;
    calculatedTargetY = targetY + nodeHeight / 2;
    calculatedTargetX = targetX + targetNodeWidth / 2;
    calculatedSourceY = sourceY - nodeHeight / 2;
    calculatedSourceX = sourceX + sourceNodeWidth / 2;
  }

  if (sourceY < targetY && sourceX > targetX) {
    calculatedSourcePosition = Position.Bottom;
    calculatedTargetPosition = Position.Right;
    calculatedSourceY = sourceY;
    calculatedTargetY = targetY + nodeHeight / 2;
    calculatedTargetX = targetX + targetNodeWidth / 2;
  }

  if (sourceY > targetY && sourceX < targetX) {
    calculatedSourcePosition = Position.Bottom;
    calculatedTargetPosition = Position.Right;
    calculatedTargetY = targetY + nodeHeight / 2;
    calculatedTargetX = targetX + targetNodeWidth / 2;
  }

  if (sourceY < targetY && sourceX < targetX) {
    calculatedSourcePosition = Position.Bottom;
    calculatedTargetPosition = Position.Left;
    calculatedTargetY = targetY + nodeHeight / 2;
    calculatedTargetX = targetX - targetNodeWidth / 2;
  }

  // This is using the actual node Y coordinates because those would tell more acurately if the nodes are in the same line
  if (
    sourceNode.metadata.y === targetNode.metadata.y &&
    calculatedSourceX < calculatedTargetX
  ) {
    calculatedSourcePosition = Position.Right;
    calculatedTargetPosition = Position.Left;
    calculatedSourceX = sourceX + sourceNodeWidth / 2;
    calculatedSourceY = sourceY - nodeHeight / 2;
    calculatedTargetX = targetX - targetNodeWidth / 2;
    calculatedTargetY = targetY + nodeHeight / 2;
  }

  if (
    sourceNode.metadata.y === targetNode.metadata.y &&
    calculatedSourceX > calculatedTargetX
  ) {
    calculatedSourcePosition = Position.Left;
    calculatedTargetPosition = Position.Right;
    calculatedSourceX = sourceX - sourceNodeWidth / 2;
    calculatedSourceY = sourceY - nodeHeight / 2;
    calculatedTargetX = targetX + targetNodeWidth / 2;
    calculatedTargetY = targetY + nodeHeight / 2;
  }

  if (
    sourceNode.metadata.x === targetNode.metadata.x &&
    calculatedSourceY > calculatedTargetY
  ) {
    calculatedSourcePosition = Position.Left;
    calculatedTargetPosition = Position.Left;
    calculatedSourceX = sourceX - sourceNodeWidth / 2;
    calculatedSourceY = sourceY - nodeHeight / 2;
    calculatedTargetX = targetX - targetNodeWidth / 2;
    calculatedTargetY = targetY + nodeHeight / 2;
  }

  return {
    calculatedSourceX,
    calculatedSourceY,
    calculatedTargetX,
    calculatedTargetY,
    calculatedTargetPosition,
    calculatedSourcePosition,
  };
}

export const CustomEdge = (props: EdgeProps) => {
  const {
    source,
    sourceX,
    sourceY,
    target,
    targetX,
    targetY,
    markerEnd,
    selected,
  } = props;
  const { nodes } = useCanvasContext();
  const { getToken } = useTheme();
  const defaultEdgeColor = getToken("colors.border.emphasized") as string;
  const highlightedEdgeColor = getToken("colors.fg") as string;
  const edgeColor = selected ? highlightedEdgeColor : defaultEdgeColor;
  const edgeBorderRadius = parseInt(getToken("radii.lg") as string);

  const sourceNode = nodes.find((n) => n.id === source);
  const targetNode = nodes.find((n) => n.id === target);

  if (!sourceNode || !targetNode) {
    return null;
  }
  const {
    calculatedSourceX,
    calculatedSourceY,
    calculatedTargetX,
    calculatedTargetY,
    calculatedSourcePosition,
    calculatedTargetPosition,
  } = getEdgeCoordinatesAndPosition({
    sourceNode: sourceNode.data,
    targetNode: targetNode.data,
    sourceX,
    sourceY,
    targetX,
    targetY,
  });

  // We need a custom algorithm to calculate the best coordinates
  const [edgePath] = getSmoothStepPath({
    sourceX: calculatedSourceX,
    sourceY: calculatedSourceY,
    targetX: calculatedTargetX,
    targetY: calculatedTargetY,
    sourcePosition: calculatedSourcePosition,
    targetPosition: calculatedTargetPosition,
    borderRadius: edgeBorderRadius,
  });

  return (
    <BaseEdge
      path={edgePath}
      markerEnd={markerEnd}
      style={{
        stroke: edgeColor,
      }}
    />
  );
};
