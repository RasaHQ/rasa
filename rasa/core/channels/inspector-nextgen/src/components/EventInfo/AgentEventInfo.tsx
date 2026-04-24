import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
import {
  type ConversationEvent
} from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface AgentEventInfoProps {
  event: ConversationEvent;
  onClose: () => void;
}

export const AgentEventInfo = ({
  event,
  onClose,
}: AgentEventInfoProps) => {
  const agentName =
    event.metadata?.agent_id ?? event.name ?? "agent";
  const flowId = event.flowId || event.metadata?.active_flow;
  const description = event.metadata?.description || "-";

  return (
    <DetailView title="Sub-agent event details" onClose={onClose}>
      <Box>
        <Heading size="md" mb="0.25rem">
          Name
        </Heading>
        <Text size="md" mb="0.5rem" wordBreak="break-all">
          {agentName}
        </Text>
      </Box>

      <Box>
        <Heading size="md" mt="0.5rem" mb="0.25rem">
          Description
        </Heading>
        <Text size="md" mb="0.5rem">
          {description}
        </Text>
      </Box>

      {flowId && (
        <Box>
          <Heading size="md" mt="0.5rem" mb="0.25rem">
            Trigger
          </Heading>
          <Text size="md" mb="0.5rem" wordBreak="break-all">
            This sub-agent was triggered by flow {flowId}
          </Text>
        </Box>
      )}

      <Separator mt="0.5rem" />

      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event info" value="event-info">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
