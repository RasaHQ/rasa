import { Flex } from "@chakra-ui/react";
import {
  ArrowRightArrowLeft,
  ArrowUpRightFromSquare,
  CheckToSlot,
  Code,
  CodeMerge,
  Comment,
  CommentXMark,
  Icon as IconComponent,
  Question,
  Robot,
  Square,
  Wrench,
} from "../../Icon";
import { CallType, FlowNodeType, type FlowNode } from "../../types";

interface Props {
  node: FlowNode;
}

const getColorPalette = (node: FlowNode): string => {
  switch (node.type) {
    case FlowNodeType.CollectInformation:
      return "green";
    case FlowNodeType.Message:
      return "blue";
    case FlowNodeType.CustomAction:
      return "orange";
    case FlowNodeType.SetSlots:
    case FlowNodeType.Start:
      return "cyan";
    case FlowNodeType.Condition:
    case FlowNodeType.Logic:
      return "pink";
    case undefined:
      return "red";
    case FlowNodeType.Call: {
      if (node.callType === CallType.Flow) return "gray";
      return "purple";
    }
    case FlowNodeType.Link:
    default:
      return "gray";
  }
};

export const Icon = ({ node }: Props) => {
  const colorPalette = getColorPalette(node);
  return (
    <Flex
      colorPalette={colorPalette}
      borderRadius="lg"
      bg="colorPalette.solid"
      color="colorPalette.contrast"
      w="1.875rem"
      h="1.875rem"
      minWidth="1.875rem"
      justifyContent="center"
      alignItems="center"
    >
      <IconComponent icon={getIcon(node)} size="lg" />
    </Flex>
  );
};

const getIcon = (node: FlowNode) => {
  switch (node.type) {
    case FlowNodeType.CollectInformation:
      return Question;
    case FlowNodeType.Message:
      return Comment;
    case FlowNodeType.CustomAction:
      return Code;
    case FlowNodeType.Condition:
    case FlowNodeType.Logic:
      return CodeMerge;
    case FlowNodeType.SetSlots:
      return CheckToSlot;
    case FlowNodeType.Call: {
      if (node.callType === CallType.Flow) {
        return ArrowRightArrowLeft;
      } else if (node.callType === CallType.Agent) {
        return Robot;
      }
      return Wrench;
    }
    case FlowNodeType.Start:
      return Square;
    case undefined:
      return CommentXMark;
    case FlowNodeType.Link:
    default:
      return ArrowUpRightFromSquare;
  }
};
