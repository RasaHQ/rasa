import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
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
        <Heading textStyle="sm" mb="1">
          Name
        </Heading>
        <Text textStyle="sm" mb="2" wordBreak="break-all">
          {flow?.name || event.flowId || "-"}
        </Text>
      </Box>

      <Box>
        <Heading textStyle="sm" mt="2" mb="1">
          Description
        </Heading>
        <Text textStyle="sm" mb="2">
          {flow?.description || "-"}
        </Text>
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
