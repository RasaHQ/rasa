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
    px: "4",
    alignItems: "center",
    justifyContent: "flex-start",
    width: "100%",
    position: "relative",
    minWidth: data.type !== FlowNodeType.Logic ? "11.5rem" : undefined, // same width as in Canvas.tsx
  };

  const textSx = {
    ml: "3",
    wordBreak: "break-word",
    display: "-webkit-box",
    "-webkit-line-clamp": "3",
    "-webkit-box-orient": "vertical",
    overflow: "hidden",
  };

  return (
    <Flex css={containerSx}>
      <NodeIcon node={data} />
      <Text textStyle="xs" css={textSx}>
        {extractNodeLabel(data)}
      </Text>
    </Flex>
  );
};
