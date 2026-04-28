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

export const Icon = ({ node }: Props) => {
  const type = node.type;
  const bgColor = getBackgroundColor(node);
  const color = getColor(type);
  const containerSx = {
    borderRadius: "0.5rem",
    color: color[0],
    bg: bgColor[0],
    w: "1.875rem",
    h: "1.875rem",
    justifyContent: "center",
    alignItems: "center",
  };
  return (
    <Flex css={containerSx}>
      <IconComponent icon={getIcon(node)} size="lg" />
    </Flex>
  );
};

const getColor = (type: FlowNodeType | "invalid") => {
  if (type === FlowNodeType.Start) {
    return ["rasaCyan.800"];
  }
  return ["rasaNeutral.50"];
};

const getBackgroundColor = (node: FlowNode) => {
  switch (node.type) {
    case FlowNodeType.CollectInformation:
      return ["rasaGreen.800"];
    case FlowNodeType.Message:
      return ["rasaBlue.900"];
    case FlowNodeType.CustomAction:
      return ["rasaOrange.600"];
    case FlowNodeType.SetSlots:
    case FlowNodeType.Start:
      return ["rasaCyan.800"];
    case FlowNodeType.Condition:
    case FlowNodeType.Logic:
      return ["rasaPink.900"];
    case undefined:
      return ["rasaRed.800"];
    case FlowNodeType.Call: {
      if (node.callType === CallType.Flow) {
        return ["rasawebDeepPurple.800"];
      }
      return ["rasawebLavender.700"];
    }
    case FlowNodeType.Link:
    default:
      return ["rasawebDeepPurple.800"];
  }
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
