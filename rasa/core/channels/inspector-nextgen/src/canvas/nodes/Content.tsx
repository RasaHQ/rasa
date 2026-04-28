import { Flex, Text } from "@chakra-ui/react";
import type { Node } from "reactflow";
import { extractNodeLabel } from "../../utils";
import { Icon as NodeIcon } from "./Icon";
import { type FlowNode, FlowNodeType } from "../../types";

interface Props {
  node: Node<FlowNode>;
}

export const Content = ({ node }: Props) => {
  const { data } = node;
  const containerSx = {
    flexGrow: 1,
    px: "1rem",
    alignItems: "center",
    justifyContent: "flex-start",
    width: "100%",
    position: "relative",
    minWidth: data.type !== FlowNodeType.Logic ? "11.5rem" : undefined, // same width as in Canvas.tsx
  };

  const textSx = {
    ml: "0.75rem",
    wordBreak: "break-word",
    display: "-webkit-box",
    "-webkit-line-clamp": "3",
    "-webkit-box-orient": "vertical",
    overflow: "hidden",
  };

  return (
    <Flex css={containerSx}>
      <NodeIcon node={data} />
      <Text size="xs" css={textSx}>
        {extractNodeLabel(data)}
      </Text>
    </Flex>
  );
};
