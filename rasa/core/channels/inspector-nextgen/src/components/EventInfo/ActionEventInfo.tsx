import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
import type { ConversationEvent } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";
import { ErrorAlert } from "../../try-assistant/ConversationLog/Event/ErrorAlert";

interface ActionEventInfoProps {
  event: ConversationEvent;
  onClose: () => void;
}

export const ActionEventInfo = ({ event, onClose }: ActionEventInfoProps) => {
  const didActionFail = event.metadata?.execution_success === false;
  const failureReason = event.metadata?.execution_error_message;

  return (
    <DetailView title="Action event details" onClose={onClose}>
      <Box data-testid="action-event-info">
        <Heading textStyle="sm" mb="1">
          Name
        </Heading>
        <Text textStyle="sm" mb="2">
          {event.name || "-"}
        </Text>

        {didActionFail && (
          <ErrorAlert
            title={failureReason
              ? "This custom action failed to execute due to the following reason:"
              : "This custom action failed to execute"}
            message={failureReason}
          />
        )}
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
