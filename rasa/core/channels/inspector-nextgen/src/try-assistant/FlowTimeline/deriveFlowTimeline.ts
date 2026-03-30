import { type UnionEventType, ConversationEventType } from "../../types";
import { isInternalRasaFlow } from "../../utils";
import type { FlowTimelineEntry } from "./types";

/**
 * Derives a timeline of flow invocations from conversation events.
 * Returns entries in reverse chronological order (newest first).
 *
 * @param events - Flat list of conversation events (across all conversations)
 * @param flowNames - Optional map of flowId → display name for human-readable labels
 */
export function deriveFlowTimeline(
  events: UnionEventType[],
  flowNames?: Map<string, string>,
): FlowTimelineEntry[] {
  const entries: FlowTimelineEntry[] = [];
  const openFlows = new Map<string, FlowTimelineEntry[]>();

  for (const event of events) {
    if (event.__typename !== "ConversationEvent") continue;

    const convEvent = event;
    const { flowId } = convEvent;

    if (!flowId || isInternalRasaFlow(flowId)) {
      continue;
    }

    switch (convEvent.conversationEventType) {
      case ConversationEventType.FlowStarted: {
        const entry: FlowTimelineEntry = {
          id: convEvent.id,
          flowId,
          flowName: flowNames?.get(flowId),
          status: "active",
          startTime: new Date(convEvent.timestamp),
        };
        const stack = openFlows.get(flowId) ?? [];
        stack.push(entry);
        openFlows.set(flowId, stack);
        entries.push(entry);
        break;
      }
      case ConversationEventType.FlowCompleted:
      case ConversationEventType.FlowCancelled: {
        const stack = openFlows.get(flowId);
        if (stack?.length) {
          const entry = stack.pop()!;
          entry.status =
            convEvent.conversationEventType ===
            ConversationEventType.FlowCompleted
              ? "completed"
              : "cancelled";
          entry.endTime = new Date(convEvent.timestamp);
          if (stack.length === 0) openFlows.delete(flowId);
        }
        break;
      }
      case ConversationEventType.FlowInterrupted: {
        const stack = openFlows.get(flowId);
        if (stack?.length) {
          stack[stack.length - 1].status = "interrupted";
        }
        break;
      }
      case ConversationEventType.FlowResumed: {
        const stack = openFlows.get(flowId);
        if (stack?.length) {
          stack[stack.length - 1].status = "active";
        }
        break;
      }
    }
  }

  return entries.slice().reverse();
}
