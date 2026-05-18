import { Box, Flex, Text } from "@chakra-ui/react";
import React, { forwardRef } from "react";
import type { ConversationEvent, ConversationEventAction } from "../../../types";
import { isInternalRasaEvent } from "../../../utils";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";

interface PlainEventProps {
  event: ConversationEvent;
  conversationEventActions?: ConversationEventAction[];
}

export const PlainEvent = forwardRef<HTMLDivElement | null, PlainEventProps>(
  (props: PlainEventProps, ref) => {
    const { event, conversationEventActions, ...otherProps } = props;
    const [isHovered, setIsHovered] = React.useState(false);

    const eventSx = {
      mb: "2",
      ml: "6",
      _last: { mb: 0 },
    };

    const containerSxBot = {
      padding: "2",
      pr: `3.5rem`,
      _first: { mt: 0 },
    };

    if (isInternalRasaEvent(event.conversationEventType.toLowerCase())) {
      return null;
    }

    return (
      <Flex
        css={containerSxBot}
        ref={ref}
        position="relative"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        {...otherProps}
      >
        {isHovered && <ConversationEventActionButton event={event} actions={conversationEventActions} />}
        <Box ml="8">
          <Text textStyle="sm" variant="muted" css={eventSx}>
            {event.conversationEventType.toLowerCase()}
          </Text>
        </Box>
      </Flex>
    );
  },
);
