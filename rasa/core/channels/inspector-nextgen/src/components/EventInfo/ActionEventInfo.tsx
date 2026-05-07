import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
import type { ConversationEvent } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

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
        <Heading size="md" mb="1">
          Name
        </Heading>
        <Text size="md" mb="2">
          {event.name || "-"}
        </Text>

        {didActionFail && (
          <Box
            bg="rasaRed.50"
            borderLeft="3px solid"
            borderColor="red.500"
            p="0.5rem"
            mb="0.5rem"
            borderRadius="0.25rem"
          >
            <Heading size="sm">
              {failureReason
                ? "This custom action failed to execute due to the following reason:"
                : "This custom action failed to execute"}
            </Heading>
            {failureReason && (
              <Text
                size="sm"
                mt="0.25rem"
                wordBreak="break-all"
              >
                {failureReason}
              </Text>
            )}
          </Box>
        )}
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
