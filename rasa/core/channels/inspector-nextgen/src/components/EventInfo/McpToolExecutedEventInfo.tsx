import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
import { RasaCodeBlock } from "../../RasaCodeBlock";
import { ErrorAlert } from "../../try-assistant/ConversationLog/Event/ErrorAlert";
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
      <Box mb="2">
        <Heading textStyle="sm" mb="1">
          Tool
        </Heading>
        <Text textStyle="sm">{toolName}</Text>
      </Box>

      {event.flowId && (
        <Box mb="2">
          <Heading textStyle="sm" mb="1">
            Flow
          </Heading>
          <Text textStyle="sm">{event.flowId}</Text>
        </Box>
      )}

      <Heading textStyle="sm" mb="1">
        Arguments
      </Heading>
      <Box maxHeight="48" overflowY="auto" borderRadius="lg" mb="2">
        <RasaCodeBlock
          code={argumentsValue != null ? JSON.stringify(argumentsValue, null, 2) : "—"}
          data-testid="event-mcp-tool-arguments"
        />
      </Box>

      {isError ? (
        <ErrorAlert
          title={errorMessage ?
            "This tool call failed to execute due to the following reason:" :
            "This tool call failed to execute"}
          message={errorMessage}
        />
      ) : (
        <>
          <Heading textStyle="sm" mb="1">
            Result
          </Heading>
          <Box
            maxHeight="80"
            overflowY="auto"
            borderRadius="lg"
            mb="2"
          >
            <RasaCodeBlock
              code={resultValue != null
                ? typeof resultValue === "string"
                  ? resultValue
                  : JSON.stringify(resultValue, null, 2)
                : "—"}
              data-testid="event-mcp-tool-result"
            />
          </Box>
        </>
      )}

      <Separator mt="2" />
      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo event={event} />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
