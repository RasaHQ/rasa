import { Accordion, Box, Heading, Separator, Text } from "@chakra-ui/react";
import type { Utterance } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";
import { BotLatencyAccordionItem } from "./LatencyDetails";
import { botUtteranceHasLatencyMetadata } from "../../utils/latency";
import { RasaCodeBlock } from "../../RasaCodeBlock";

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
          <Heading textStyle="sm" mb="1">
            Name
          </Heading>
          <Text textStyle="sm" mb="2" wordBreak="break-all">
            {utterAction}
          </Text>
        </Box>
      )}

      {utterance.rephrase && (
        <>
          <Box>
            <Heading textStyle="sm" mb="1">
              Original response
            </Heading>
            <Text textStyle="sm" mb="2">
              {utterance.metadata?.domain_ground_truth ?? "-"}
            </Text>
          </Box>

          <Box>
            <Heading textStyle="sm" mb="1">
              Rephrased response
            </Heading>
            <Text textStyle="sm" mb="2">
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
            <RasaCodeBlock
              code={JSON.stringify(utterance.responseData.custom, null, 2)}
            />
          </EventAccordionItem>
        )}
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo
            event={{ ...utterance, conversationEventType: "BOT" }}
          />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView  >
  );
};
