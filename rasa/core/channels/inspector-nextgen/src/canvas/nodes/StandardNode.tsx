import { type NodeProps, Handle, Position } from "reactflow";
import { SharedBox } from "./SharedBox";
import { nodePropsToNodeObject } from "../../utils";
import { Content } from "./Content";

export const StandardNode = (props: NodeProps) => {
  const { isConnectable, targetPosition, sourcePosition, selected } = props;
  const node = nodePropsToNodeObject(props);
  const boxSx: {
    pointerEvents: string;
    borderColor: string;
    _hover?: { boxShadow: string; cursor: string };
  } = {
    pointerEvents: "all",
    borderColor: selected ? "rasawebDeepPurple.800" : "rasaNeutral.500",
  };

  return (
    <>
      <Handle
        type="target"
        position={targetPosition || Position.Top}
        isConnectable={false}
        style={{
          top: -0.25,
        }}
      />
      <SharedBox tabIndex={0} css={boxSx} data-testid="node">
        <Content node={node} />
      </SharedBox>

      <Handle
        type="source"
        position={sourcePosition || Position.Bottom}
        isConnectable={isConnectable}
        style={{
          bottom: 0,
        }}
      />

      <Handle
        type="source"
        position={Position.Left}
        isConnectable={isConnectable}
      />
      <Handle
        type="target"
        position={Position.Right}
        isConnectable={isConnectable}
      />
      <Handle
        type="target"
        position={Position.Left}
        isConnectable={isConnectable}
      />
      <Handle
        type="source"
        position={Position.Right}
        isConnectable={isConnectable}
      />
    </>
  );
};
