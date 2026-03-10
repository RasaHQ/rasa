import { Accordion, Box, Code, Separator, Text } from "@chakra-ui/react";
import type { ConversationEvent } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface SlotEventInfoProps {
  event: ConversationEvent;
  onClose: () => void;
}

export const SlotEventInfo = ({ event, onClose }: SlotEventInfoProps) => {
  return (
    <DetailView title="Slot event" onClose={onClose}>
      <Box data-testid="event-slot-or-flow-name">
        <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
          Slot name
        </Text>
        <Text fontSize="0.875rem" mb="0.5rem">
          {event.name || "-"}
        </Text>
      </Box>

      <Box data-testid="event-slot-value">
        <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
          Value
        </Text>
        <Code
          whiteSpace="pre-wrap"
          display="block"
          fontSize="0.75rem"
          p="1rem"
          mb="0.5rem"
          borderRadius="0.5rem"
          variant="solid"
          fontFamily="IBM Plex Mono"
        >
          {JSON.stringify(event.slotValue, null, 2)}
        </Code>
      </Box>

      <Separator mt="0.5rem" />

      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
