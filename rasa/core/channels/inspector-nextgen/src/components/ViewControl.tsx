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

  const containerStyles = {
    xl: {
      position: "absolute !important",
      left: "50%",
      translate: "-50%",
    },
  };

  return (
    <Box css={containerStyles}>
      <SegmentGroup.Root
        data-testid="view-control"
        size="sm"
        defaultValue={inspectMode ? "inspect" : "chat"}
        onValueChange={(details) => {
          setInspectMode(details.value === "inspect");
        }}
      >
        <SegmentGroup.Indicator />
        <SegmentGroup.Items items={items} />
      </SegmentGroup.Root>
    </Box>
  );
};
