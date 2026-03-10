import { Accordion, Box, Separator, Text } from "@chakra-ui/react";
import type { ConversationEvent, Flow } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface FlowEventInfoProps {
  event: ConversationEvent;
  flow?: Flow;
  onClose: () => void;
}

export const FlowEventInfo = ({ event, flow, onClose }: FlowEventInfoProps) => {
  return (
    <DetailView title="Flow event" onClose={onClose}>
      <Box data-testid="event-slot-or-flow-name">
        <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
          Name
        </Text>
        <Text fontSize="0.875rem" mb="0.5rem" wordBreak="break-all">
          {flow?.name || event.flowId || "-"}
        </Text>
      </Box>

      <Box>
        <Text fontWeight="bold" fontSize="0.875rem" mt="0.5rem" mb="0.25rem">
          Description
        </Text>
        <Text fontSize="0.875rem" mb="0.5rem">
          {flow?.description || "-"}
        </Text>
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
