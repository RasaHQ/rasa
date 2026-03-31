import { Box, Flex } from "@chakra-ui/react";
import { InspectorViewHeader } from "../components/InspectorViewHeader";
import { NoData } from "../Placeholder";
import { PlaceholderImage } from "../types";
import { useInspectorStore } from "../store";
import { SlotsDetails } from "./MemoryDetails";

export const MemorySection = () => {
  const slotRelatedEvents = useInspectorStore((s) => s.slotRelatedEvents);

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
        <InspectorViewHeader title="Collected slots" />
      </Box>
      <Box flex={1} overflowY="auto" bg="white">
        {slotRelatedEvents.length > 0 ? (
          <SlotsDetails />
        ) : (
          <NoData
            image={PlaceholderImage.Cubes}
            shortLabel="Collected slots"
            longLabel="Collected slots will be shown here."
          />
        )}
      </Box>
    </Flex>
  );
};
