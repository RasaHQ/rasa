import { Text } from "@chakra-ui/react";
import { type NodeProps, Handle, Position } from "reactflow";
import type { FlowNodeStart } from "../../types";
import { extractNodeLabel } from "../../utils";

export const StartAloneNode = (props: NodeProps<FlowNodeStart>) => {
  const { data, isConnectable, sourcePosition } = props;

  const sx = {
    bg: "teal.solid",
    color: "fg.inverted",
    borderRadius: "full",
    border: "none",
    borderColor: "transparent",
    boxShadow: "tooltip",
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
      />
    </>
  );
};
