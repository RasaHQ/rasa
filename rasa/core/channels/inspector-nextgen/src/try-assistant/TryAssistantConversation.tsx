import { useEffect, useRef } from "react";
import { Box } from "@chakra-ui/react";
import type { Conversation, ConversationEventAction, UnionEventType } from "../types";
import { ConversationSession } from "./ConversationSession";

interface Props {
  conversationList: Conversation[];
  conversationAssistant?: Record<string, string>;
  onQuickReply?: (payload: string) => void;
  onSelect?: (selection: UnionEventType) => void;
  selectedElementId?: string;
  inspectorMode: boolean;
  replayConversation?: (eventId: string) => void;
  waitingForResponse?: boolean;
  replayingConversation?: boolean;
  conversationEventActions?: ConversationEventAction[];
}

export const TryAssistantConversation = ({
  conversationList,
  onQuickReply,
  conversationAssistant = {},
  onSelect,
  selectedElementId,
  inspectorMode,
  replayConversation,
  conversationEventActions,
  waitingForResponse = false,
  replayingConversation = false,
}: Props) => {
  const scrollToHighlightedMessageRef = useRef<HTMLDivElement>(null);

  const lastConversation = conversationList[conversationList.length - 1];
  const lastEventId =
    lastConversation?.events[lastConversation.events.length - 1]?.id;
  const lastConversationId = lastConversation?.id;

  // When a new conversation starts (restart), reset scroll positions on all
  // scrollable ancestors so stale offsets don't clip the fresh content.
  useEffect(() => {
    let el = scrollToHighlightedMessageRef.current?.parentElement;
    while (el) {
      if (el.scrollTop !== 0) {
        el.scrollTop = 0;
      }
      el = el.parentElement;
    }
  }, [lastConversationId]);

  // Scroll to the last event in the conversation. Also triggers when
  // inspectorMode is toggled since that changes the number of visible events.
  useEffect(() => {
    setTimeout(() => {
      scrollToHighlightedMessageRef.current?.scrollIntoView?.({
        behavior: "smooth",
        block: "center",
      });
    }, 0);
  }, [lastEventId, inspectorMode]);

  const nonEmptyConversationList = conversationList.filter(
    (conversation, index) =>
      conversation.totalNumberOfUserMessages > 0 ||
      index === conversationList.length - 1,
  );

  return (
    <Box data-testid="assistant-chat">
      {nonEmptyConversationList.map((conversation, index) => (
        <ConversationSession
          interactive={index === nonEmptyConversationList.length - 1}
          key={conversation.id}
          conversation={conversation}
          onQuickReply={onQuickReply}
          assistantVersion={conversationAssistant[conversation.id]}
          selectable={true}
          onSelect={onSelect}
          replayConversation={replayConversation}
          selectedElementId={selectedElementId}
          inspectorMode={inspectorMode}
          waitingForResponse={
            index === nonEmptyConversationList.length - 1 &&
            waitingForResponse && nonEmptyConversationList[index].totalNumberOfUserMessages > 0
          }
          replayingConversation={
            index === nonEmptyConversationList.length - 1 &&
            replayingConversation
          }
          conversationEventActions={conversationEventActions}
        />
      ))}
      <div ref={scrollToHighlightedMessageRef} />
    </Box>
  );
};
