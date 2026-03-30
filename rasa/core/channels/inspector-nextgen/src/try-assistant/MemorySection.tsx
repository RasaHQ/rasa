import { Box } from "@chakra-ui/react";
import { InspectorViewHeader } from "../components/InspectorViewHeader";
import { NoData } from "../Placeholder";
import { PlaceholderImage } from "../types";

export const MemorySection = () => (
  <Box position="relative" width="100%" height="100%">
    <InspectorViewHeader title="Memory" />
    <NoData
      image={PlaceholderImage.Cubes}
      shortLabel="Memory"
      longLabel="Assistant memory will be shown here."
    />
  </Box>
);
