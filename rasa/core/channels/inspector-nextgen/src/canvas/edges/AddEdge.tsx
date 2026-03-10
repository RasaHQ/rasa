import { type EdgeProps, BaseEdge, getSmoothStepPath } from "reactflow";
import { useCanvasContext } from "../../CanvasContext";

export const AddEdge = (props: EdgeProps) => {
  const {
    source,
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    style = {},
    markerEnd,
  } = props;
  const { nodes } = useCanvasContext();
  const [edgePath] = getSmoothStepPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
    borderRadius: 0,
  });
  const sourceNode = nodes.find((n) => n.id === source);

  if (!sourceNode) return null;

  return <BaseEdge path={edgePath} markerEnd={markerEnd} style={style} />;
};
