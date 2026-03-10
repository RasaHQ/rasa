import { Text } from "@chakra-ui/react";
import { type NodeProps, Handle, Position } from "reactflow";
import { extractNodeLabel } from "../../utils";
import { type FlowNode } from "../../types";

export const StartNode = (props: NodeProps<FlowNode>) => {
  const { data, isConnectable, sourcePosition } = props;

  const sx = {
    bg: "#2C9793",
    color: "#FFFFFF",
    borderRadius: "9999px",
    border: "none",
    borderColor: "transparent",
    px: "1.5rem",
    py: "0.5rem",
    pointerEvents: "all",
    height: "auto",
  };

  return (
    <>
      <Text size="xs" tabIndex={0} css={sx} data-testid="node">
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
