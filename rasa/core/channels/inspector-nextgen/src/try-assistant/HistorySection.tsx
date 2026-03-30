import { Box, Flex } from "@chakra-ui/react";
import { useCallback, useMemo } from "react";
import { InspectorViewHeader } from "../components/InspectorViewHeader";
import { NoData } from "../Placeholder";
import { useInspectorStore } from "../store";
import { toggleSelectedElement } from "../store/actions";
import { PlaceholderImage } from "../types";
import { deriveFlowTimeline, FlowTimeline } from "./FlowTimeline";

export const HistorySection = () => {
  const conversationList = useInspectorStore((s) => s.conversationList);
  const flows = useInspectorStore((s) => s.flows);
  const allEvents = useMemo(
    () => conversationList.flatMap((c) => c.events),
    [conversationList],
  );

  const entries = useMemo(() => {
    const flowNames = new Map(flows.map((f) => [f.id, f.name ?? f.id]));
    return deriveFlowTimeline(allEvents, flowNames);
  }, [allEvents, flows]);

  const handleEntryClick = useCallback(
    (entryId: string) => {
      const event = allEvents.find((e) => e.id === entryId);
      if (event) toggleSelectedElement(event);
    },
    [allEvents],
  );

  return (
    <Flex
      position="relative"
      direction="column"
      width="100%"
      height="100%"
      overflow="hidden"
      borderTopRightRadius="xl"
      borderBottomRightRadius="xl"
    >
      <Box
        flexShrink={0}
        position="relative"
        zIndex={1}
        bg="white"
        borderTopRightRadius="xl"
      >
        <InspectorViewHeader title="Flow history" />
      </Box>
      <Box flex={1} overflowY="auto" bg="white">
        {entries.length > 0 ? (
          <FlowTimeline entries={entries} onEntryClick={handleEntryClick} />
        ) : (
          <NoData
            image={PlaceholderImage.Cubes}
            shortLabel="History"
            longLabel="Conversation history will be shown here."
          />
        )}
      </Box>
    </Flex>
  );
};
