import { Accordion, Box, Code, Heading, Separator, Text } from "@chakra-ui/react";
import type { ConversationEvent } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface McpToolExecutedEventInfoProps {
  event: ConversationEvent;
  onClose: () => void;
}

type RawMcpEvent = {
  tool_name?: string;
  arguments?: Record<string, unknown>;
  result?: unknown;
  is_error?: boolean;
  error_message?: string;
};

export const McpToolExecutedEventInfo = ({
  event,
  onClose,
}: McpToolExecutedEventInfoProps) => {
  const rawEvent = event.metadata?.rawEvent as RawMcpEvent | undefined;
  const toolName =
    event.metadata?.tool_name ?? rawEvent?.tool_name ?? event.name ?? "tool";
  const argumentsValue =
    event.metadata?.tool_arguments ?? rawEvent?.arguments ?? null;
  const resultValue = event.metadata?.tool_result ?? rawEvent?.result;
  const isError =
    event.metadata?.tool_is_error === true || rawEvent?.is_error === true;
  const errorMessage =
    event.metadata?.tool_error_message ?? rawEvent?.error_message;

  return (
    <DetailView title="MCP tool executed" onClose={onClose}>
      <Box mb="0.5rem">
        <Heading size="md" mb="0.25rem">
          Tool
        </Heading>
        <Text size="md">{toolName}</Text>
      </Box>

      {event.flowId && (
        <Box mb="0.5rem">
          <Heading size="md" mb="0.25rem">
            Flow
          </Heading>
          <Text size="md">{event.flowId}</Text>
        </Box>
      )}

      <Heading size="md" mb="0.25rem">
        Arguments
      </Heading>
      <Box maxHeight="12rem" overflowY="auto" borderRadius="0.5rem" mb="0.5rem">
        <Code
          whiteSpace="pre-wrap"
          display="block"
          fontSize="0.75rem"
          p="1rem"
          borderRadius="0.5rem"
          variant="solid"
          fontFamily="IBM Plex Mono"
          data-testid="event-mcp-tool-arguments"
        >
          {argumentsValue != null ? JSON.stringify(argumentsValue, null, 2) : "—"}
        </Code>
      </Box>

      {isError ? (
        <Box
          bg="red.50"
          borderLeft="3px solid"
          borderColor="red.500"
          p="0.5rem"
          mb="0.5rem"
          borderRadius="0.25rem"
        >
          <Heading size="sm">
            {errorMessage
              ? "This tool call failed to execute due to the following reason:"
              : "This tool call failed to execute"}
          </Heading>
          {errorMessage && (
            <Text
              size="sm"
              mt="0.25rem"
            >
              {errorMessage}
            </Text>
          )}
        </Box>
      ) : (
        <>
          <Heading size="md" mb="0.25rem">
            Result
          </Heading>
          <Box
            maxHeight="20rem"
            overflowY="auto"
            borderRadius="0.5rem"
            mb="0.5rem"
          >
            <Code
              whiteSpace="pre-wrap"
              display="block"
              fontSize="0.75rem"
              p="1rem"
              borderRadius="0.5rem"
              variant="solid"
              fontFamily="IBM Plex Mono"
              data-testid="event-mcp-tool-result"
            >
              {resultValue != null
                ? typeof resultValue === "string"
                  ? resultValue
                  : JSON.stringify(resultValue, null, 2)
                : "—"}
            </Code>
          </Box>
        </>
      )}

      <Separator mt="0.5rem" />
      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
