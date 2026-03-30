import { Box, Flex, Text } from "@chakra-ui/react";
import type { FlowInvocationStatus, FlowTimelineEntry } from "./types";

const STATUS_CONFIG: Record<
  FlowInvocationStatus,
  { dotColor: string; tagBg: string; tagColor: string; label: string }
> = {
  active: {
    dotColor: "rasaGreen.800",
    tagBg: "rasaGreen.50",
    tagColor: "rasaGreen.700",
    label: "Active",
  },
  interrupted: {
    dotColor: "rasaYellow.500",
    tagBg: "rasaYellow.50",
    tagColor: "rasaYellow.900",
    label: "Interrupted",
  },
  completed: {
    dotColor: "rasaNeutral.500",
    tagBg: "rasaNeutral.300",
    tagColor: "rasaNeutral.800",
    label: "Completed",
  },
  cancelled: {
    dotColor: "rasaNeutral.500",
    tagBg: "rasaNeutral.300",
    tagColor: "rasaNeutral.800",
    label: "Cancelled",
  },
};

const CONNECTOR_COLOR = "rasaNeutral.400";

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

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
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
  const config = STATUS_CONFIG[entry.status];

  const subtitle = entry.endTime
    ? `Started at ${formatTime(entry.startTime)}  \u2022  Duration: ${formatDuration(entry.startTime, entry.endTime)}`
    : `Started at ${formatTime(entry.startTime)}`;

  return (
    <Flex
      gap="4px"
      data-testid="flow-timeline-item"
      cursor={onClick ? "pointer" : undefined}
      borderRadius="md"
      _hover={onClick ? { bg: "rasaNeutral.100" } : undefined}
      onClick={onClick ? () => onClick(entry.id) : undefined}
    >
      {/* Timeline gutter: top connector → dot → bottom connector */}
      <Flex
        direction="column"
        alignItems="center"
        width="24px"
        flexShrink={0}
      >
        {isFirst ? (
          <Box height="16px" flexShrink={0} />
        ) : (
          <Box
            width="1px"
            height="16px"
            bg={CONNECTOR_COLOR}
            flexShrink={0}
          />
        )}
        <Box
          width="8px"
          height="8px"
          borderRadius="full"
          bg={config.dotColor}
          flexShrink={0}
        />
        {isLast ? (
          <Box flex={1} />
        ) : (
          <Box width="1px" flex={1} bg={CONNECTOR_COLOR} />
        )}
      </Flex>

      {/* Content row: flow info + status tag */}
      <Flex
        py="8px"
        pr="8px"
        flex={1}
        justifyContent="space-between"
        alignItems="flex-start"
        minWidth={0}
      >
        <Flex direction="column" minWidth={0}>
          <Text
            size="md"
            fontWeight="500"
            lineHeight="1.7"
            truncate
          >
            {entry.flowName ? capitalize(entry.flowName) : entry.flowId}
          </Text>
          <Text size="xs" variant="muted" whiteSpace="pre">
            {subtitle}
          </Text>
        </Flex>

        <Flex
          bg={config.tagBg}
          px="10px"
          py="4px"
          borderRadius="full"
          alignItems="center"
          justifyContent="center"
          flexShrink={0}
          ml="8px"
        >
          <Text
            fontSize="12px"
            fontWeight="500"
            lineHeight="1.5"
            letterSpacing="0.4px"
            color={config.tagColor}
          >
            {config.label}
          </Text>
        </Flex>
      </Flex>
    </Flex>
  );
}
