import { Box, HStack, Separator, Text } from "@chakra-ui/react";
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

  const headingSx = {
    color: "rasaNeutral.700",
    fontSize: "0.75rem",
    textAlign: "center",
    maxWidth: "80%",
  };

  const versionSx = {
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  };

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
    <Box mt="1rem" mb="1rem">
      <HStack padding="0.5rem" width="100%" justifyContent="center">
        <Separator orientation="horizontal" marginLeft={"1rem"} />
        <Box css={headingSx} flex="0 1 auto">
          <Text size="sm" variant="muted">
            Session started on {formattedStartDate}
          </Text>
          {assistantVersion ? (
            <Text css={versionSx}>{assistantVersion}</Text>
          ) : null}
        </Box>
        <Separator orientation="horizontal" marginRight={"1rem"} />
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
    </Box>
  );
};
