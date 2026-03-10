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
  Square,
} from "../../Icon";
import { FlowNodeType } from "../../types";

interface Props {
  type: FlowNodeType | "invalid";
}

export const Icon = ({ type }: Props) => {
  const bgColor = getBackgroundColor(type);
  const color = getColor(type);
  const containerSx = {
    borderRadius: "0.5rem",
    color: color[0],
    p: "0.25rem",
    bg: bgColor[0],
    width: "20px",
    justifyContent: "center",
  };
  return (
    <Flex css={containerSx}>
      <IconComponent icon={getIcon(type)} />
    </Flex>
  );
};

const getColor = (type: FlowNodeType | "invalid") => {
  if (type === FlowNodeType.Start) {
    return ["rasaCyan.800"];
  }
  return ["rasaNeutral.50"];
};

const getBackgroundColor = (type: FlowNodeType | "invalid") => {
  switch (type) {
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
    case "invalid":
      return ["rasaRed.800"];
    case FlowNodeType.Link:
    case FlowNodeType.Call:
    default:
      return ["rasawebDeepPurple.800"];
  }
};

const getIcon = (type: FlowNodeType | "invalid") => {
  switch (type) {
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
    case FlowNodeType.Call:
      return ArrowRightArrowLeft;
    case FlowNodeType.Start:
      return Square;
    case "invalid":
      return CommentXMark;
    case FlowNodeType.Link:
    default:
      return ArrowUpRightFromSquare;
  }
};
