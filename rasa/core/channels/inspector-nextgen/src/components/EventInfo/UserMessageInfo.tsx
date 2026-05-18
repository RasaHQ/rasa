import { Accordion, Box, Heading, HStack, Separator, Text } from "@chakra-ui/react";
import { RasaCodeBlock } from "../../RasaCodeBlock";
import type { Utterance } from "../../types";
import { CommonEventInfo } from "./CommonEventInfo";
import { DetailView } from "./DetailView";
import { EventAccordionItem } from "./EventAccordionItem";

interface UserMessageInfoProps {
  utterance: Utterance;
  onClose: () => void;
}

export const UserMessageInfo = ({ utterance, onClose }: UserMessageInfoProps) => {
  const getConfidencePalette = (confidence: number) =>
    confidence > 0.7 ? "green" : confidence > 0.4 ? "yellow" : "red";

  const intentPills = utterance.intents?.slice(0, 3).map((intent) => (
    <HStack key={intent.id} mb="1">
      <Box
        colorPalette={getConfidencePalette(intent.confidence)}
        px="1.5"
        py="0.5"
        borderRadius="sm"
        fontWeight="600"
      >
        {(intent.confidence * 100).toFixed(1)}%
      </Box>
      <Text>{intent.name}</Text>
    </HStack>
  ));

  return (
    <DetailView title="User message details" onClose={onClose}>
      <Heading textStyle="sm" mt="2" mb="1">
        Predicted intents
      </Heading>
      {intentPills && intentPills.length > 0 ? (
        <Box mb="2">{intentPills}</Box>
      ) : (
        <Text mb="2">-</Text>
      )}

      {utterance.commands ? (
        <>
          <Heading textStyle="sm" mt="2" mb="1">
            Predicted Commands
          </Heading>

          <RasaCodeBlock
            code={utterance.commands != null ? JSON.stringify(utterance.commands, null, 2) || "-" : "—"}
          />
        </>
      ) : null}

      <Separator mt="2" />

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
