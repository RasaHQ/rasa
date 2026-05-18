import { Box, HStack, SegmentGroup } from "@chakra-ui/react";
import { Comment, Icon, LocationCrosshairs } from "../Icon";
import { useInspectorStore } from "../store";
import { setInspectMode } from "../store/actions";

const views = [
  { label: "Chat", icon: Comment, value: "chat" },
  { label: "Inspect", icon: LocationCrosshairs, value: "inspect" },
];

export const ViewControl = () => {
  const inspectMode = useInspectorStore((s) => s.inspectMode);

  const items = views.map(({ value, icon, label }) => ({
    value,
    label: (
      <HStack>
        <Icon icon={icon} />
        {label}
      </HStack>
    ),
  }));

  return (
    <Box>
      <SegmentGroup.Root
        data-testid="view-control"
        size="sm"
        defaultValue={inspectMode ? "inspect" : "chat"}
        onValueChange={(details) => {
          setInspectMode(details.value === "inspect");
        }}
        css={{ "--segment-indicator-bg": "colors.bg" }}
      >
        <SegmentGroup.Indicator />
        <SegmentGroup.Items items={items} />
      </SegmentGroup.Root>
    </Box >
  );
};
