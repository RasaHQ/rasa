import { Flex } from "@chakra-ui/react";
import { useCallback, useMemo } from "react";
import { InspectorViewHeader } from "../components/InspectorViewHeader";
import { ScrollFadeArea } from "../components/ScrollFadeArea";
import { NoData } from "../Placeholder";
import { useInspectorStore, toggleSelectedElement } from "../store";
import { PlaceholderImage } from "../types";
import { deriveFlowTimeline, FlowTimeline } from "./FlowTimeline";

interface HistorySectionProps {
  showViewSwitcher?: boolean;
}

export const HistorySection = ({ showViewSwitcher }: HistorySectionProps) => {
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
      direction="column"
      width="100%"
      height="100%"
      overflow="hidden"
    >
      <InspectorViewHeader
        title="Flow history"
        showViewSwitcher={showViewSwitcher}
      />
      <ScrollFadeArea>
        {entries.length > 0 ? (
          <FlowTimeline entries={entries} onEntryClick={handleEntryClick} />
        ) : (
          <NoData
            image={PlaceholderImage.Cubes}
            shortLabel="History"
            longLabel="Conversation history will be shown here."
          />
        )}
      </ScrollFadeArea>
    </Flex>
  );
};
