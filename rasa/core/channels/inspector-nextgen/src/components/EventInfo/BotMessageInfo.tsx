import { Accordion, Box, Code, Separator, Text } from "@chakra-ui/react";
import type { Utterance } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface BotMessageInfoProps {
  utterance: Utterance;
  onClose: () => void;
}

export const BotMessageInfo = ({ utterance, onClose }: BotMessageInfoProps) => {
  const utterAction = utterance.metadata?.utter_action;

  return (
    <DetailView title="Agent response details" onClose={onClose}>
      {utterAction && (
        <Box>
          <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
            Name
          </Text>
          <Text fontSize="0.875rem" mb="0.5rem" wordBreak="break-all">
            {utterAction}
          </Text>
        </Box>
      )}

      {utterance.rephrase && (
        <>
          <Box>
            <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
              Original response
            </Text>
            <Text fontSize="0.875rem" mb="0.5rem">
              {utterance.metadata?.domain_ground_truth ?? "-"}
            </Text>
          </Box>

          <Box>
            <Text fontWeight="bold" fontSize="0.875rem" mb="0.25rem">
              Rephrased response
            </Text>
            <Text fontSize="0.875rem" mb="0.5rem">
              {utterance.text ?? "-"}
            </Text>
          </Box>
        </>
      )}

      <Separator mt="0.5rem" />

      <Accordion.Root collapsible multiple>
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
