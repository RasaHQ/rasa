import { Accordion, Box, Code, Heading, Separator, Text } from "@chakra-ui/react";
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
        <Heading size="md" mb="1">
          Slot name
        </Heading>
        <Text size="md" mb="2">
          {event.name || "-"}
        </Text>
      </Box>

      <Box data-testid="event-slot-value">
        <Heading size="md" mb="1">
          Value
        </Heading>
        <Code
          whiteSpace="pre-wrap"
          display="block"
          fontSize="0.75rem"
          p="1rem"
          mb="2"
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
