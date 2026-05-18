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
    <Box bg="white" borderRadius="lg">
      <ScrollContainer>
        <ScrollContent withSpacing={false} pl="4" pr="7" py="2" css={{ "& [data-orientation='vertical']": { display: "none" } }}>
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
    <Box display="flex" flexDirection="column" gap="4">
      {sections.map((section, index) => (
        <Box key={index}>
          {index > 0 && (
            <Separator mb="4" mx="-4" />
          )}
          <Box display="flex" flexDirection="column" data-testid={`${section.title}-section`}>
            <Heading
              textStyle="sm"
              px="2"
              py="1"
              mb="1"
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
      px="2"
      py="1"
      borderRadius="lg"
      bg={isActive ? "bg.muted" : "transparent"}
      color="fg"
      _hover={{
        bg: "bg.muted",
        cursor: "pointer",
      }}
      onClick={() => slot.event ? handleSlotClick(slot.event) : undefined}
      data-testid={`slot-${slot.name}`}
    >
      <Text
        variant="primary"
        textStyle="sm"
        width="50%"
        truncate
        title={slot.name}
      >
        {slot.name}
      </Text>
      <Text
        variant="primary"
        textStyle="sm"
        fontFamily="mono"
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
