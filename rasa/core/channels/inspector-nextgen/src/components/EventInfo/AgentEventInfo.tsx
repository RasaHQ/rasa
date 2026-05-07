import { Accordion, Box, Code, Heading, Separator, Tag, Text } from "@chakra-ui/react";
import { useCallback } from "react";
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
    event.agentId ?? event.name ?? "agent";
  const flowId = event.flowId || event.metadata?.active_flow;
  const description = event.metadata?.description || "-";
  const tools = event.metadata?.mcp_tools || [];
  const excludedTools = event.metadata?.excluded_mcp_tools || [];
  const exitConditions = event.metadata?.exit_conditions;

  const toolTagStyles = {
    mr: "0.25rem",
    maxWidth: "20rem",
  };

  const renderTools = useCallback((tools: string[]) => (
    <Box mb="0.5rem" lineHeight="2rem">
      {tools.map(tool => (
        <Tag.Root key={tool} size="lg" variant="subtle" rounded="full" css={toolTagStyles}>
          <Tag.Label>{tool}</Tag.Label>
        </Tag.Root>
      ))}
    </Box>
    // ignoring because it's not important :)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  ), []);

  return (
    <DetailView title="Sub-agent event details" onClose={onClose}>
      <Box>
        <Heading size="md" mb="1">
          Name
        </Heading>
        <Text size="md" mb="2" wordBreak="break-all">
          {agentName}
        </Text>
      </Box>

      <Box>
        <Heading size="md" mt="2" mb="1">
          Description
        </Heading>
        <Text size="md" mb="2">
          {description}
        </Text>
      </Box>

      {!!tools.length && (
        <Box>
          <Heading size="md" mt="2" mb="1">
            Tools
          </Heading>
          {renderTools(tools)}
        </Box>
      )}

      {!!excludedTools.length && (
        <Box>
          <Heading size="md" mt="2" mb="1">
            Excluded tools
          </Heading>
          {renderTools(excludedTools)}
        </Box>
      )}

      {!tools.length && !excludedTools.length && (
        <Box>
          <Heading size="md" mt="2" mb="1">
            Tools
          </Heading>
          <Text size="md" mb="0.5rem" wordBreak="break-all">
            All server tools available
          </Text>
        </Box>
      )}

      {flowId && (
        <Box>
          <Heading size="md" mt="2" mb="1">
            Trigger
          </Heading>
          <Text size="md" mb="2" wordBreak="break-all">
            This sub-agent was triggered by flow {flowId}
          </Text>
        </Box>
      )}

      <Separator mt="2" />

      {exitConditions && (
        <Accordion.Root collapsible multiple>
          <EventAccordionItem title="Exit conditions" value="exit-conditions">
            <Code
              whiteSpace="pre-wrap"
              display="block"
              fontSize="0.75rem"
              p="1rem"
              borderRadius="0.5rem"
              variant="solid"
              fontFamily="IBM Plex Mono"
              data-testid="agent-exit-conditions"
            >
              {exitConditions.map((condition: string) => (
                `- ${condition}`
              )).join("\n")}
            </Code>
          </EventAccordionItem>
        </Accordion.Root>
      )}

      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event info" value="event-info">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
