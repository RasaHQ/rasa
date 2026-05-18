import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
import { RasaCodeBlock } from "../../RasaCodeBlock";
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
        <Heading textStyle="sm" mb="1">
          Slot name
        </Heading>
        <Text textStyle="sm" mb="2">
          {event.name || "-"}
        </Text>
      </Box>

      <Box data-testid="event-slot-value">
        <Heading textStyle="sm" mb="1">
          Value
        </Heading>
        <RasaCodeBlock
          code={JSON.stringify(event.slotValue, null, 2)}
          data-testid="event-mcp-tool-result"
        />
      </Box>

      <Separator mt="2" />

      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
