import { useEffect, useRef } from "react";
import type { Conversation, ConversationEventAction, UnionEventType } from "../types";
import {
  ScrollContainer,
  ScrollContent,
} from "../VerticalScroll";
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

  //If there is no utterance to scroll to, scroll to the last header/utterance
  const lastConversation = conversationList[conversationList.length - 1];
  const lastEventId =
    lastConversation?.events[lastConversation.events.length - 1]?.id;

  // scrolls to the last event in the conversation, also triggers when
  // inspectorMode is toggled since that changes the number of events in the
  // conversation list and we need to scroll to the bottom again.

  useEffect(() => {
    // The setTimeout is needed to make the `scrollIntoView` happen at
    // the next tick. Without it, `scrollIntoView` doesn't seem to work
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
    <ScrollContainer data-testid="assistant-chat">
      <ScrollContent withSpacing={false}>
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
              waitingForResponse
            }
            replayingConversation={
              index === nonEmptyConversationList.length - 1 &&
              replayingConversation
            }
            conversationEventActions={conversationEventActions}
          />
        ))}
        <div ref={scrollToHighlightedMessageRef} />
      </ScrollContent>
    </ScrollContainer>
  );
};
