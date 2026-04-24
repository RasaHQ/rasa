import {
  type ConversationEvent,
  ConversationEventType,
  type Flow,
  type UnionEventType,
  UtteranceType,
} from "../../types";
import { isUtterance } from "../../utils";
import { ActionEventInfo } from "./ActionEventInfo";
import { AgentEventInfo } from "./AgentEventInfo";
import { BotMessageInfo } from "./BotMessageInfo";
import { FlowEventInfo } from "./FlowEventInfo";
import { McpToolExecutedEventInfo } from "./McpToolExecutedEventInfo";
import { PlainEventInfo } from "./PlainEventInfo";
import { SlotEventInfo } from "./SlotEventInfo";
import { UserMessageInfo } from "./UserMessageInfo";

function isFlowEvent(event: ConversationEvent): boolean {
  return [
    ConversationEventType.FlowCancelled,
    ConversationEventType.FlowCompleted,
    ConversationEventType.FlowInterrupted,
    ConversationEventType.FlowResumed,
    ConversationEventType.FlowStarted,
  ].includes(event.conversationEventType);
}

function isSlotEvent(event: ConversationEvent): boolean {
  return [
    ConversationEventType.ResetSlots,
    ConversationEventType.Slot,
  ].includes(event.conversationEventType);
}

function isActionEvent(event: ConversationEvent): boolean {
  return event.conversationEventType === ConversationEventType.Action;
}

function isMcpToolExecutedEvent(event: ConversationEvent): boolean {
  return event.conversationEventType === ConversationEventType.McpToolExecuted;
}

function isAgentEvent(event: ConversationEvent): boolean {
  return [
    ConversationEventType.AgentStarted,
    ConversationEventType.AgentCompleted,
    ConversationEventType.AgentCancelled,
    ConversationEventType.AgentInterrupted,
    ConversationEventType.AgentResumed,
  ].includes(event.conversationEventType);
}

interface EventInfoProps {
  event: UnionEventType;
  onClose: () => void;
  flows: Flow[];
}

export const EventInfo = ({ event, onClose, flows }: EventInfoProps) => {
  if (isUtterance(event) && event.type === UtteranceType.Bot) {
    return <BotMessageInfo utterance={event} onClose={onClose} />;
  }

  if (isUtterance(event)) {
    return <UserMessageInfo utterance={event} onClose={onClose} />;
  }

  if (isFlowEvent(event)) {
    const flow = flows.find((f) => f.id === event.flowId || f.name === event.flowId);
    return <FlowEventInfo event={event} flow={flow} onClose={onClose} />;
  }

  if (isSlotEvent(event)) {
    return <SlotEventInfo event={event} onClose={onClose} />;
  }

  if (isActionEvent(event)) {
    return <ActionEventInfo event={event} onClose={onClose} />;
  }

  if (isMcpToolExecutedEvent(event)) {
    return <McpToolExecutedEventInfo event={event} onClose={onClose} />;
  }

  if (isAgentEvent(event)) {
    return <AgentEventInfo event={event} onClose={onClose} />;
  }

  return <PlainEventInfo event={event} onClose={onClose} />;
};
