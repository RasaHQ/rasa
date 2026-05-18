import { type NodeProps, Handle, Position } from "reactflow";
import { SharedBox } from "./SharedBox";
import { Content } from "./Content";
import { nodePropsToNodeObject } from "../../utils";

export const EndNode = (props: NodeProps) => {
  const { targetPosition, sourcePosition, selected } = props;
  const node = nodePropsToNodeObject(props);

  const boxSx: {
    pointerEvents: string;
    overflow: string;
    borderColor?: string;
    _hover?: { boxShadow: string; cursor: string };
  } = {
    pointerEvents: "all",
    overflow: "visible",
    borderColor: selected ? "fg" : "border.emphasized",
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
      <Handle
        type="source"
        position={sourcePosition || Position.Bottom}
        isConnectable={false}
        style={{
          bottom: 0,
        }}
      />
      <Handle type="source" position={Position.Left} isConnectable={false} />
      <Handle type="target" position={Position.Right} isConnectable={false} />
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <Handle type="source" position={Position.Right} isConnectable={false} />

      <SharedBox tabIndex={0} css={boxSx} data-testid="node">
        <Content node={node} />
      </SharedBox>
    </>
  );
};
