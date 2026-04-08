import { Flex } from "@chakra-ui/react";
import { InspectorViewHeader } from "../components/InspectorViewHeader";
import { ScrollFadeArea } from "../components/ScrollFadeArea";
import { NoData } from "../Placeholder";
import { PlaceholderImage } from "../types";
import { useInspectorStore } from "../store";
import { SlotsDetails } from "./MemoryDetails";

interface MemorySectionProps {
  showViewSwitcher?: boolean;
}

export const MemorySection = ({ showViewSwitcher }: MemorySectionProps) => {
  const slotRelatedEvents = useInspectorStore((s) => s.slotRelatedEvents);

  return (
    <Flex direction="column" width="100%" height="100%">
      <InspectorViewHeader
        title="Collected slots"
        showViewSwitcher={showViewSwitcher}
      />
      <ScrollFadeArea>
        {slotRelatedEvents.length > 0 ? (
          <SlotsDetails />
        ) : (
          <NoData
            image={PlaceholderImage.Cubes}
            shortLabel="Collected slots"
            longLabel="Collected slots will be shown here."
          />
        )}
      </ScrollFadeArea>
    </Flex>
  );
};
