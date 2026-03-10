import { Accordion, Box, Code, HStack, Separator, Text } from "@chakra-ui/react";
import type { Utterance } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface UserMessageInfoProps {
  utterance: Utterance;
  onClose: () => void;
}

export const UserMessageInfo = ({ utterance, onClose }: UserMessageInfoProps) => {
  const intentPills = utterance.intents?.slice(0, 3).map((intent) => (
    <HStack key={intent.id} mb="0.25rem">
      <Box
        bg={intent.confidence > 0.7 ? "green.100" : intent.confidence > 0.4 ? "yellow.100" : "red.100"}
        color={intent.confidence > 0.7 ? "green.800" : intent.confidence > 0.4 ? "yellow.800" : "red.800"}
        px="0.375rem"
        py="0.125rem"
        borderRadius="0.25rem"
        fontSize="0.688rem"
        fontWeight="600"
      >
        {(intent.confidence * 100).toFixed(1)}%
      </Box>
      <Text fontSize="0.813rem">{intent.name}</Text>
    </HStack>
  ));

  return (
    <DetailView title="User message details" onClose={onClose}>
      <Text fontWeight="bold" fontSize="0.75rem" mt="0.5rem" mb="0.25rem">
        Predicted intents
      </Text>
      {intentPills && intentPills.length > 0 ? (
        <Box mb="0.5rem">{intentPills}</Box>
      ) : (
        <Text fontSize="0.813rem" mb="0.5rem">-</Text>
      )}

      {utterance.commands ? (
        <>
          <Text fontWeight="bold" fontSize="0.75rem" mt="0.5rem" mb="0.25rem">
            Predicted Commands
          </Text>
          <Code whiteSpace="pre-wrap" display="block" fontSize="0.75rem" p="0.5rem" mb="0.5rem">
            {JSON.stringify(utterance.commands, null, 2) || "-"}
          </Code>
        </>
      ) : null}

      <Separator mt="0.5rem" />

      <Accordion.Root collapsible multiple>
        <EventAccordionItem title="Event details" value="event-details">
          <CommonEventInfo
            event={{ ...utterance, conversationEventType: "USER" }}
          />
        </EventAccordionItem>
      </Accordion.Root>
    </DetailView>
  );
};
