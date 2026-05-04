import {
  type ConversationEvent,
  type UnionEventType,
  ConversationEventType,
} from "../../types";
import { isConversationEvent, isInternalRasaFlow } from "../../utils";
import type { FlowTimelineEntry } from "./types";
import { orderBy } from "lodash";

const isFlowEvent = (event: ConversationEvent) => {
  return [
    ConversationEventType.FlowStarted,
    ConversationEventType.FlowInterrupted,
    ConversationEventType.FlowResumed,
    ConversationEventType.FlowCancelled,
    ConversationEventType.FlowCompleted
  ].includes(event.conversationEventType);
}

const isAgentEvent = (event: ConversationEvent) => {
  return [
    ConversationEventType.AgentStarted,
    ConversationEventType.AgentInterrupted,
    ConversationEventType.AgentResumed,
    ConversationEventType.AgentCancelled,
    ConversationEventType.AgentCompleted
  ].includes(event.conversationEventType) && event.agentId;
}

const mapEventToStatus = (event: ConversationEvent) => {
  switch (event.conversationEventType) {
    case ConversationEventType.FlowInterrupted:
    case ConversationEventType.AgentInterrupted:
      return "interrupted" as const;
    case ConversationEventType.FlowCancelled:
    case ConversationEventType.AgentCancelled:
      return "cancelled" as const;
    case ConversationEventType.FlowCompleted:
    case ConversationEventType.AgentCompleted:
      return "completed" as const;
    case ConversationEventType.FlowStarted:
    case ConversationEventType.AgentStarted:
    case ConversationEventType.FlowResumed:
    case ConversationEventType.AgentResumed:
    default:
      return "active" as const;
  }
}

const isStartEvent = (event: ConversationEvent) => {
  return [
    ConversationEventType.FlowStarted,
    ConversationEventType.AgentStarted
  ].includes(event.conversationEventType);
}

const isFinishedEvent = (event: ConversationEvent) => {
  return [
    ConversationEventType.FlowCompleted,
    ConversationEventType.FlowCancelled,
    ConversationEventType.AgentCompleted,
    ConversationEventType.AgentCancelled,
  ].includes(event.conversationEventType);
}

const isLastInvocationOpen = (allInvokationsForAgent: ConversationEvent[][]) => {
  const lastInvokationForAgent = allInvokationsForAgent?.at(-1);
  const lastLoggedEventForAgent = lastInvokationForAgent?.at(-1);
  return lastLoggedEventForAgent && isAgentEvent(lastLoggedEventForAgent) && !isFinishedEvent(lastLoggedEventForAgent);
}

/**
 * Derives a timeline of flow invocations from conversation events.
 * Returns entries in reverse chronological order (newest first).
 *
 * @param events - Flat list of conversation events (across all conversations)
 * @param flowNames - Optional map of flowId → display name for human-readable labels
 */
export function deriveFlowTimeline(
  events?: UnionEventType[],
  flowNames?: Map<string, string>,
): FlowTimelineEntry[] {
  const entries: FlowTimelineEntry[] = [];
  const invocationsByFlowOrAgentId: { [id: string]: ConversationEvent[][] } = {};

  if (!events) {
    return [];
  }

  for (const convEvent of events) {
    if (!isConversationEvent(convEvent)) {
      continue;
    }

    const { flowId, agentId } = convEvent;
    const key = agentId || flowId;
    if ((isFlowEvent(convEvent) && !isInternalRasaFlow(key)) || isAgentEvent(convEvent)) {
      if (isStartEvent(convEvent)) {
        if (isAgentEvent(convEvent) && isLastInvocationOpen(invocationsByFlowOrAgentId[key])) {
          continue;
        }
        invocationsByFlowOrAgentId[key] = [...(invocationsByFlowOrAgentId[key] || []), [convEvent]];
      } else {
        const invocations = invocationsByFlowOrAgentId[key];
        // TODO: check that it works for calling flows from themselves
        invocations?.at(-1)?.push(convEvent);
      }
    }
  }

  for (const key in invocationsByFlowOrAgentId) {
    for (const invocationEvents of invocationsByFlowOrAgentId[key]) {
      if (!invocationEvents?.length) continue;
      const sortedEvents = orderBy(invocationEvents, "timestamp");
      const firstEvent = sortedEvents.at(0);
      const lastEvent = sortedEvents.at(-1);
      if (!lastEvent || !firstEvent) continue;
      const { flowId } = firstEvent;
      entries.push({
        id: firstEvent.id,
        type: isFlowEvent(firstEvent) ? "flow" : "agent",
        flowId,
        flowName: flowNames?.get(flowId),
        agentId: isAgentEvent(firstEvent) ? key : undefined,
        status: mapEventToStatus(lastEvent),
        startTime: new Date(firstEvent.timestamp),
        exactStartTime: firstEvent.originalTimestamp,
        endTime: isFinishedEvent(lastEvent) ? new Date(lastEvent.timestamp) : undefined,
      });
    }
  }

  return orderBy(entries, "exactStartTime", "desc");
}
