import { type FlexProps } from "@chakra-ui/react";
import { forwardRef } from "react";
import {
  type ConversationEvent,
  type ConversationEventAction,
  ConversationEventType
} from "../../../types";
import { ActionEvent } from "./ActionEvent";
import { FlowEvent } from "./FlowEvent";
import { PlainEvent } from "./PlainEvent";
import { SlotEvent } from "./SlotEvent";

const flowEventTypes = [
  ConversationEventType.FlowCancelled,
  ConversationEventType.FlowCompleted,
  ConversationEventType.FlowStarted,
  ConversationEventType.FlowInterrupted,
  ConversationEventType.FlowResumed,
];
const slotEventTypes = [
  ConversationEventType.Slot,
  ConversationEventType.ResetSlots,
];

interface Props extends FlexProps {
  event: ConversationEvent;
  selectable?: boolean;
  isSelected?: boolean;
  onEventSelect?: (selection: ConversationEvent) => void;
  replayConversation?: (eventId: string) => void;
  isLastActiveEvent?: boolean;
  conversationEventActions?: ConversationEventAction[];
}

export const Event = forwardRef<HTMLDivElement | null, Props>(
  (props: Props, ref) => {
    const {
      event,
      selectable = false,
      onEventSelect,
      isSelected = false,
      replayConversation,
      isLastActiveEvent = false,
      ...otherProps
    } = props;
    const handleSelect = () => {
      if (selectable && onEventSelect) {
        onEventSelect(event);
      }
    };

    const additionalProps = {
      ...otherProps,
      "data-testid": isLastActiveEvent
        ? "conversation-events-last-active-event"
        : "conversation-events",
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "Enter" && selectable && onEventSelect) {
        onEventSelect(event);
      }
    };

    if (flowEventTypes.includes(event.conversationEventType)) {
      return (
        <FlowEvent
          ref={ref}
          event={event}
          onClick={handleSelect}
          onKeyDown={handleKeyDown}
          isSelected={selectable && isSelected}
          {...additionalProps}
        />
      );
    }

    if (slotEventTypes.includes(event.conversationEventType)) {
      return (
        <SlotEvent
          ref={ref}
          event={event}
          onClick={handleSelect}
          onKeyDown={handleKeyDown}
          isSelected={selectable && isSelected}
          {...additionalProps}
        />
      );
    }

    if (event.conversationEventType === ConversationEventType.Action) {
      return (
        <ActionEvent
          ref={ref}
          event={event}
          onClick={handleSelect}
          onKeyDown={handleKeyDown}
          isSelected={selectable && isSelected}
          replayConversation={replayConversation}
          {...additionalProps}
        />
      );
    }

    return <PlainEvent event={event} ref={ref} {...additionalProps} />;
  },
);
