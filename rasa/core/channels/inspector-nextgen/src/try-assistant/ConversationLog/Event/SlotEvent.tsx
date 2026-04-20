import { Box, Flex, Text } from "@chakra-ui/react";
import React, { forwardRef } from "react";
import { Icon } from "../../../Icon";
import { CheckToSlot } from "../../../Icon/icons";
import {
  type ConversationEvent,
  type ConversationEventAction,
  ConversationEventType
} from "../../../types";
import { isInternalRasaSlot } from "../../../utils";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";
import { useConversationLogSx } from "../useConversationLogSx";

function generateSlotEventMessage(event: ConversationEvent) {
  if (event.conversationEventType === ConversationEventType.ResetSlots) {
    return <Text variant="muted">Slots reset</Text>;
  }

  const isSlotCleared =
    (event.conversationEventType === ConversationEventType.Slot &&
      event.metadata?.reset) ||
    event.slotValue === null;

  const suffix = isSlotCleared ? "is cleared" : "set";

  const modifiedSuffix = event.metadata?.was_modified_by_studio
    ? " (modified)"
    : "";

  return (
    <Text size="sm" lineClamp={2} wordBreak="break-all" variant="muted">
      Slot{" "}
      <Text as="span" fontWeight="500" variant="muted">
        {event.name}
      </Text>{" "}
      {suffix}
      {modifiedSuffix}
    </Text>
  );
}

interface SlotEventProps {
  event: ConversationEvent;
  isSelected: boolean;
  conversationEventActions?: ConversationEventAction[];
}

export const SlotEvent = forwardRef<HTMLDivElement | null, SlotEventProps>(
  (props: SlotEventProps, ref) => {
    const { event, isSelected = false, conversationEventActions, ...otherProps } = props;
    const { hoverableContainerSx, baseMessageSx, iconSx } =
      useConversationLogSx(isSelected);
    const [isHovered, setIsHovered] = React.useState(false);

    if (isInternalRasaSlot(event.name || "")) {
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
          <Icon icon={CheckToSlot} style={iconSx} />
          {generateSlotEventMessage(event)}
        </Box>
      </Flex>
    );
  },
);
