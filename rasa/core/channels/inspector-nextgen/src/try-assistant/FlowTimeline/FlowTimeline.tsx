import { Flex } from "@chakra-ui/react";
import type { FlowTimelineEntry } from "./types";
import { FlowTimelineItem } from "./FlowTimelineItem";

interface FlowTimelineProps {
  readonly entries: FlowTimelineEntry[];
  readonly onEntryClick?: (entryId: string) => void;
}

/**
 * Renders a vertical timeline of flow invocations during a conversation.
 * Newest entries appear at the top. Each entry shows the flow name,
 * start time, optional duration, and a status tag.
 *
 * Use `deriveFlowTimeline` to build entries from conversation events.
 */
export function FlowTimeline({ entries, onEntryClick }: FlowTimelineProps) {
  if (entries.length === 0) return null;

  return (
    <Flex direction="column" bg="white" px="16px" py="8px" data-testid="flow-timeline">
      {entries.map((entry, index) => (
        <FlowTimelineItem
          key={entry.id}
          entry={entry}
          isFirst={index === 0}
          isLast={index === entries.length - 1}
          onClick={onEntryClick}
        />
      ))}
    </Flex>
  );
}
