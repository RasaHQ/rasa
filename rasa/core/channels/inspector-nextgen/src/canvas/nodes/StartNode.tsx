import { Text } from "@chakra-ui/react";
import { type NodeProps, Handle, Position } from "reactflow";
import { extractNodeLabel } from "../../utils";
import { type FlowNode } from "../../types";

export const StartNode = (props: NodeProps<FlowNode>) => {
  const { data, isConnectable, sourcePosition } = props;

  const sx = {
    bg: "teal.solid",
    color: "fg.inverted",
    borderRadius: "full",
    border: "none",
    borderColor: "transparent",
    px: "6",
    py: "2",
    pointerEvents: "all",
    height: "auto",
  };

  return (
    <>
      <Text textStyle="xs" tabIndex={0} css={sx} data-testid="node">
        {extractNodeLabel(data)}
      </Text>
      <Handle
        type="source"
        position={sourcePosition || Position.Bottom}
        isConnectable={isConnectable}
        style={{
          bottom: 0,
        }}
      />
    </>
  );
};
