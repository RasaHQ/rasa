import { Accordion, Box, Code, Heading, Separator, Text } from "@chakra-ui/react";
import type { Utterance } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";
import { BotLatencyAccordionItem } from "./LatencyDetails";
import { botUtteranceHasLatencyMetadata } from "../../utils/latency";

interface BotMessageInfoProps {
  utterance: Utterance;
  onClose: () => void;
}

export const BotMessageInfo = ({ utterance, onClose }: BotMessageInfoProps) => {
  const utterAction = utterance.metadata?.utter_action;
  const showLatency = botUtteranceHasLatencyMetadata(utterance);

  return (
    <DetailView title="Agent response details" onClose={onClose}>
      {utterAction && (
        <Box>
          <Heading size="md" mb="1">
            Name
          </Heading>
          <Text size="md" mb="2" wordBreak="break-all">
            {utterAction}
          </Text>
        </Box>
      )}

      {utterance.rephrase && (
        <>
          <Box>
            <Heading size="md" mb="1">
              Original response
            </Heading>
            <Text size="md" mb="2">
              {utterance.metadata?.domain_ground_truth ?? "-"}
            </Text>
          </Box>

          <Box>
            <Heading size="md" mb="1">
              Rephrased response
            </Heading>
            <Text size="md" mb="2">
              {utterance.text ?? "-"}
            </Text>
          </Box>
        </>
      )}

      <Separator mt="2" />

      <Accordion.Root
        collapsible
        multiple
      >
        {showLatency && (
          <BotLatencyAccordionItem
            executionTimes={utterance.metadata?.execution_times}
            voiceLatency={utterance.metadata?.voiceLatency}
          />
        )}
        {utterance.responseData?.custom && (
          <EventAccordionItem title="Custom response" value="custom-response">
            <Code
              whiteSpace="pre-wrap"
              display="block"
              fontSize="0.75rem"
              p="0.5rem"
            >
              {JSON.stringify(utterance.responseData.custom, null, 2)}
            </Code>
          </EventAccordionItem>
        )}
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo
            event={{ ...utterance, conversationEventType: "BOT" }}
          />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
