import { Accordion, Box, Heading, Separator, Tag, Text } from "@chakra-ui/react";
import { useCallback } from "react";
import { RasaCodeBlock } from "../../RasaCodeBlock";
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
    mr: "1",
    maxWidth: "80",
  };

  const renderTools = useCallback((tools: string[]) => (
    <Box mb="2">
      {tools.map(tool => (
        <Tag.Root key={tool} size="xl" variant="subtle" rounded="full" css={toolTagStyles}>
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
        <Heading textStyle="sm" mb="1">
          Name
        </Heading>
        <Text textStyle="sm" mb="2" wordBreak="break-all">
          {agentName}
        </Text>
      </Box>

      <Box>
        <Heading textStyle="sm" mt="2" mb="1">
          Description
        </Heading>
        <Text textStyle="sm" mb="2">
          {description}
        </Text>
      </Box>

      {!!tools.length && (
        <Box>
          <Heading textStyle="sm" mt="2" mb="1">
            Tools
          </Heading>
          {renderTools(tools)}
        </Box>
      )}

      {!!excludedTools.length && (
        <Box>
          <Heading textStyle="sm" mt="2" mb="1">
            Excluded tools
          </Heading>
          {renderTools(excludedTools)}
        </Box>
      )}

      {!tools.length && !excludedTools.length && (
        <Box>
          <Heading textStyle="sm" mt="2" mb="1">
            Tools
          </Heading>
          <Text textStyle="sm" mb="2" wordBreak="break-all">
            All server tools available
          </Text>
        </Box>
      )}

      {flowId && (
        <Box>
          <Heading textStyle="sm" mt="2" mb="1">
            Trigger
          </Heading>
          <Text textStyle="sm" mb="2" wordBreak="break-all">
            This sub-agent was triggered by flow {flowId}
          </Text>
        </Box>
      )}

      <Separator mt="2" />

      {exitConditions && (
        <Accordion.Root collapsible multiple>
          <EventAccordionItem title="Exit conditions" value="exit-conditions">
            <RasaCodeBlock
              code={exitConditions.map((condition: string) => (
                `- ${condition}`
              )).join("\n")}
              data-testid="agent-exit-conditions"
            />
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
