import { Accordion } from "@chakra-ui/react";
import type { ConversationEvent } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface PlainEventInfoProps {
  event: ConversationEvent;
  onClose: () => void;
}

export const PlainEventInfo = ({ event, onClose }: PlainEventInfoProps) => {
  return (
    <DetailView title="Event" onClose={onClose}>
      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event info" value="event-info">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
