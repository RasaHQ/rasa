import { Box, Heading, Separator, Text } from "@chakra-ui/react";
import { ScrollContainer, ScrollContent } from "../VerticalScroll";
import { toggleSelectedElement, useInspectorStore } from "../store";
import { selectLatestStack } from "../store/selectors";
import type { ConversationEvent, SlotState } from "../types";
import { extractSlotEventsForFlow, extractSlotEventsForSession, formatSlots, isSystemSlotEvent } from "../utils/conversation";

export const SlotsDetails = () => {
  const slotRelatedEvents = useInspectorStore((s) => s.slotRelatedEvents);
  const latestStack = useInspectorStore(selectLatestStack);
  const activeSlotName = latestStack?.collect;

  const currentFlowId = latestStack?.flowId;
  const currentFlowSlotEvents = currentFlowId
    ? extractSlotEventsForFlow(slotRelatedEvents, currentFlowId)
    : [];
  const sessionSlotEvents = extractSlotEventsForSession(slotRelatedEvents);
  const allSlots = formatSlots(slotRelatedEvents);

  const currentFlowSlots = formatSlots(currentFlowSlotEvents).filter(
    (slot) => slot.value
  );
  const sessionSlots = formatSlots(sessionSlotEvents).filter((sessionSlot) => {
    if (sessionSlot.value == null) {
      return false;
    }

    if (isSystemSlotEvent(sessionSlot.name)) {
      return false;
    }

    return !currentFlowSlots.some((currentFlowSlot) => {
      return currentFlowSlot.name === sessionSlot.name;
    });
  });

  const systemSlots = allSlots.filter((slot) => isSystemSlotEvent(slot.name));

  const sections = [
    ...(currentFlowSlots.length > 0
      ? [{ title: "Current flow", slots: currentFlowSlots }]
      : []),
    ...(sessionSlots.length > 0
      ? [{ title: "Session", slots: sessionSlots }]
      : []),
    ...(systemSlots.length > 0
      ? [{ title: "System", slots: systemSlots }]
      : []),
  ];

  return (
    <Box bg="white" borderRadius="0.5rem">
      <ScrollContainer>
        <ScrollContent withSpacing={false} pl="1rem" pr="1.75rem" py="0.5rem" css={{ "& [data-orientation='vertical']": { display: "none" } }}>
          <SlotSections sections={sections} activeSlotName={activeSlotName} />
        </ScrollContent>
      </ScrollContainer>
    </Box>
  );
};

interface TableSection {
  title: string;
  slots: SlotState[];
}

interface SlotSectionsProps {
  sections: TableSection[];
  activeSlotName?: string;
}

function SlotSections({ sections, activeSlotName }: SlotSectionsProps) {
  return (
    <Box display="flex" flexDirection="column" gap="1rem">
      {sections.map((section, index) => (
        <Box key={index}>
          {index > 0 && (
            <Separator mb="1rem" mx="-1rem" />
          )}
          <Box display="flex" flexDirection="column" data-testid={`${section.title}-section`}>
            <Heading
              size="md"
              px="0.5rem"
              py="0.25rem"
              mb="0.25rem"
            >
              {section.title}
            </Heading>
            {section.slots.map((slot) => (
              <SlotRow
                key={slot.name}
                slot={slot}
                isActive={slot.name === activeSlotName}
              />
            ))}
          </Box>
        </Box>
      ))}
    </Box>
  );
}

const SlotRow = ({
  slot,
  isActive,
}: {
  slot: SlotState;
  isActive: boolean;
}) => {
  const valueStr = JSON.stringify(slot.value);
  const handleSlotClick = (event: ConversationEvent) => {
    if (event) toggleSelectedElement(event);
  };

  return (
    <Box
      display="flex"
      alignItems="center"
      justifyContent="space-between"
      px="0.5rem"
      py="0.25rem"
      borderRadius="0.5rem"
      bg={isActive ? "rasaNeutral.100" : "transparent"}
      color="rasawebDeepPurple.800"
      _hover={{
        bg: "rasaNeutral.100", 
        cursor: "pointer",
      }}
      onClick={() => slot.event ? handleSlotClick(slot.event) : undefined}
      data-testid={`slot-${slot.name}`}
    >
      <Text 
        variant="primary"
        size="md"
        width="50%"
        truncate
        title={slot.name}
      >
        {slot.name}
      </Text>
      <Text
        variant="primary"
        size="sm"
        fontFamily='"IBM Plex Mono", monospace'
        width="50%"
        truncate
        title={valueStr}
        data-testid={"slot-value"}
      >
        {valueStr}
      </Text>
    </Box>
  );
};
