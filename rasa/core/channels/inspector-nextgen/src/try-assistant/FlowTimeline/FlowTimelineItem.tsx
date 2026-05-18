import { Box, Flex, Heading, Tag, Text } from "@chakra-ui/react";
import { capitalize } from "lodash";
import { useMemo } from "react";
import type { FlowInvocationStatus, FlowTimelineEntry } from "./types";

const STATUS_CONFIG: Record<FlowInvocationStatus, { palette: string; label: string }> = {
  active: { palette: "green", label: "Active" },
  interrupted: { palette: "yellow", label: "Interrupted" },
  completed: { palette: "gray", label: "Completed" },
  cancelled: { palette: "gray", label: "Cancelled" },
};

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatDuration(start: Date, end: Date): string {
  const totalSeconds = Math.round(
    (end.getTime() - start.getTime()) / 1000,
  );
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
}

interface FlowTimelineItemProps {
  readonly entry: FlowTimelineEntry;
  readonly isFirst: boolean;
  readonly isLast: boolean;
  readonly onClick?: (entryId: string) => void;
}

export function FlowTimelineItem({
  entry,
  isFirst,
  isLast,
  onClick,
}: FlowTimelineItemProps) {
  const { palette, label } = STATUS_CONFIG[entry.status];

  const title = useMemo(() => {
    if (entry.type === "agent") {
      return `Agent: ${entry.agentId ?? "unknown"}`;
    } else if (entry.flowName) {
      return capitalize(entry.flowName);
    } else {
      return entry.flowId
    }
  }, [entry.type, entry.agentId, entry.flowName, entry.flowId]);


  const subtitle = entry.endTime
    ? `Started at ${formatTime(entry.startTime)}  \u2022  Duration: ${formatDuration(entry.startTime, entry.endTime)}`
    : `Started at ${formatTime(entry.startTime)}`;

  return (
    <Flex
      colorPalette={palette}
      gap="1"
      data-testid="flow-timeline-item"
      cursor={onClick ? "pointer" : undefined}
      borderRadius="md"
      _hover={onClick ? { bg: "bg.muted" } : undefined}
      onClick={onClick ? () => onClick(entry.id) : undefined}
    >
      {/* Timeline gutter: top connector → dot → bottom connector */}
      <Flex
        direction="column"
        alignItems="center"
        width="6"
        flexShrink={0}
      >
        {isFirst ? (
          <Box height="4" flexShrink={0} />
        ) : (
          <Box
            width="px"
            height="4"
            bg="border.emphasized"
            flexShrink={0}
          />
        )}
        <Box
          width="2"
          height="2"
          borderRadius="full"
          bg="colorPalette.solid"
          flexShrink={0}
        />
        {isLast ? (
          <Box flex={1} />
        ) : (
          <Box width="px" flex={1} bg="border.emphasized" />
        )}
      </Flex>

      {/* Content row: flow info + status tag */}
      <Flex
        py="2"
        pr="2"
        flex={1}
        justifyContent="space-between"
        alignItems="flex-start"
        minWidth={0}
      >
        <Flex direction="column" minWidth={0}>
          <Heading
            textStyle="md"
            truncate
          >
            {title}
          </Heading>
          <Text textStyle="xs" variant="muted" whiteSpace="pre">
            {subtitle}
          </Text>
        </Flex>

        <Tag.Root colorPalette={palette} variant="subtle" size="lg" rounded="full">
          <Tag.Label>
            {label}
          </Tag.Label>
        </Tag.Root>
      </Flex>
    </Flex>
  );
}
