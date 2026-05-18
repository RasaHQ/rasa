import { Alert, Box, HStack } from "@chakra-ui/react";
import { useCallback } from "react";
import {
  type Conversation,
  type ConversationEventAction,
  type UnionEventType,
  UtteranceType,
} from "../types";
import {
  formatDateTime,
  isBotUtteranceEmpty,
  isConversationEvent,
  isUtterance,
} from "../utils";
import { ConversationLoadingSpinner } from "./ConversationLoadingSpinner";
import { Event as ConversationEventComponent } from "./ConversationLog/Event";
import { Message } from "./ConversationLog/Message/Message";

interface ConversationSessionProps {
  interactive?: boolean;
  conversation: Conversation;
  assistantVersion?: string;
  onQuickReply?: (payload: string) => void;
  selectable?: boolean;
  onSelect?: (selection: UnionEventType) => void;
  selectedElementId?: string;
  inspectorMode: boolean;
  replayConversation?: (eventId: string) => void;
  waitingForResponse?: boolean;
  replayingConversation?: boolean;
  conversationEventActions?: ConversationEventAction[];
}

export const ConversationSession = ({
  conversation,
  onQuickReply,
  selectable = false,
  onSelect,
  selectedElementId,
  inspectorMode,
  replayConversation,
  interactive = false,
  waitingForResponse = false,
  replayingConversation = false,
  assistantVersion,
  conversationEventActions,
}: ConversationSessionProps) => {
  const { events = [] } = conversation;

  const formattedStartDate = formatDateTime(new Date(conversation.startDate));
  const formattedEndDate = conversation.endDate ? formatDateTime(new Date(conversation.endDate)) : undefined;

  // we don't want to render stack events, becomes too noisy
  const shownEvents = events.filter(
    (event) =>
      (isUtterance(event) && !isBotUtteranceEmpty(event)) ||
      isConversationEvent(event),
  );

  const lastUtterance = shownEvents.findLast((event) => isUtterance(event));

  // only the last bot utterance can be interactive - and only if it is really
  // the last utterance and there is no user utterance after it
  const interactiveBotUtteranceId =
    lastUtterance?.type === UtteranceType.Bot ? lastUtterance.id : undefined;

  const handleSelect = useCallback(
    (selection: UnionEventType) => {
      if (inspectorMode) {
        onSelect?.(selection);
      }
    },
    [onSelect, inspectorMode],
  );

  return (
    <Box mt="4" mb="4">
      <HStack px="6" py="2" width="100%" justifyContent="center">
        <Alert.Root size="sm" justifyContent="center">
          <Alert.Title>
            Session started on {formattedStartDate} {assistantVersion || ""}
          </Alert.Title>
        </Alert.Root>
      </HStack>

      {shownEvents.map((event, index) => {
        const isSelected = event.id === selectedElementId;
        if (isUtterance(event)) {
          return (
            <Message
              key={event.id}
              utterance={event}
              isInteractive={
                interactive && event.id === interactiveBotUtteranceId
              }
              onQuickReply={onQuickReply}
              onMessageSelect={handleSelect}
              isSelected={isSelected}
              selectable={selectable}
              inspectorMode={inspectorMode}
              conversationEventActions={conversationEventActions}
            />
          );
        } else if (isConversationEvent(event)) {
          return inspectorMode ? (
            <ConversationEventComponent
              key={event.id}
              event={event}
              selectable={selectable}
              onEventSelect={handleSelect}
              isSelected={isSelected}
              replayConversation={replayConversation}
              isLastActiveEvent={
                index === shownEvents.length - 1 && interactive
              }
              conversationEventActions={conversationEventActions}
            />
          ) : null;
        }
      })}
      {waitingForResponse || replayingConversation ? (
        <ConversationLoadingSpinner />
      ) : null}

      {formattedEndDate && (
        <HStack px="6" py="2" width="100%" justifyContent="center">
          <Alert.Root size="sm" justifyContent="center">
            <Alert.Title>
              Session ended on {formattedEndDate}
            </Alert.Title>
          </Alert.Root>
        </HStack>
      )}
    </Box>
  );
};
