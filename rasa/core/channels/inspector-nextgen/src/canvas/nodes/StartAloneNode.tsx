import { Text } from "@chakra-ui/react";
import { type NodeProps, Handle, Position } from "reactflow";
import type { FlowNodeStart } from "../../types";
import { extractNodeLabel } from "../../utils";

export const StartAloneNode = (props: NodeProps<FlowNodeStart>) => {
  const { data, isConnectable, sourcePosition } = props;

  const sx = {
    bg: "#2C9793",
    color: "#FFFFFF",
    borderRadius: "9999px",
    border: "none",
    borderColor: "transparent",
    boxShadow: "0 0.125rem 0.5rem 0 rgba(0, 0, 0, 0.15)",
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
      />
    </>
  );
};
