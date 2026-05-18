import React, { forwardRef } from "react";
import {
  ConversationEventType,
  type ConversationEvent,
  type ConversationEventAction,
} from "../../../types";
import { Box, Flex, Text } from "@chakra-ui/react";
import { eventTexts, isInternalRasaFlow } from "../../../utils";
import { Icon } from "../../../Icon";
import {
  PlayCircle,
  CheckCircle,
  CircleXmark,
  ExclamationCircle,
} from "../../../Icon/icons";
import type { IconDefinition } from "@fortawesome/fontawesome-svg-core";
import { useConversationLogSx } from "../useConversationLogSx";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";

const eventTypeToIcon: Partial<Record<ConversationEventType, IconDefinition>> =
{
  [ConversationEventType.FlowStarted]: PlayCircle,
  [ConversationEventType.FlowResumed]: PlayCircle,
  [ConversationEventType.FlowCompleted]: CheckCircle,
  [ConversationEventType.FlowCancelled]: CircleXmark,
  [ConversationEventType.FlowInterrupted]: ExclamationCircle,
};

interface FlowEventProps {
  event: ConversationEvent;
  isSelected: boolean;
  conversationEventActions?: ConversationEventAction[];
}

function generateFlowEventMessage(event: ConversationEvent) {
  const prefix = event.flowId?.startsWith("pattern_") ? "System flow" : "Flow";
  const suffix = eventTexts[event.conversationEventType] ?? "";

  return (
    <Text textStyle="sm" color="fg.muted" lineClamp={2} wordBreak="break-all">
      {prefix}{" "}
      <Text as="span" fontWeight="medium" variant="muted">
        {event.flowId}
      </Text>{" "}
      {suffix}
    </Text>
  );
}

export const FlowEvent = forwardRef<HTMLDivElement | null, FlowEventProps>(
  (props: FlowEventProps, ref) => {
    const { event, isSelected = false, conversationEventActions, ...otherProps } = props;
    const { hoverableContainerSx, baseMessageSx, iconSx } =
      useConversationLogSx(isSelected);
    const [isHovered, setIsHovered] = React.useState(false);
    const iconType = eventTypeToIcon[event.conversationEventType] || null;

    if (isInternalRasaFlow(event.flowId || "")) {
      return null;
    }

    return (
      <Flex
        css={hoverableContainerSx}
        ref={ref}
        tabIndex={0}
        position="relative"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        {...otherProps}
      >
        {isHovered && <ConversationEventActionButton event={event} actions={conversationEventActions} />}
        <Box css={baseMessageSx}>
          {iconType ? <Icon icon={iconType} style={iconSx} /> : null}
          {generateFlowEventMessage(event)}
        </Box>
      </Flex>
    );
  },
);
