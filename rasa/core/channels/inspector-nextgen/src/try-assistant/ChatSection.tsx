import { Box } from "@chakra-ui/react";
import { useCallback, useRef } from "react";
import { OnboardingTooltip } from "../OnboardingTooltip";
import {
  selectWaitingForResponse,
  useInspectorStore,
  clearSelectedElement,
  setInspectMode,
} from "../store";
import { type UnionEventType } from "../types";
import {
  ScrollContainer,
  ScrollContent,
  ScrollFixedFooter,
  ScrollFixedHeader,
} from "../VerticalScroll";
import { ChatHeader } from "./ChatHeader";
import { MessageInput } from "./MessageInput";
import { TryAssistantConversation } from "./TryAssistantConversation";

interface ChatSectionProps {
  handleSelect: (selection: UnionEventType) => void;
}

export const ChatSection = ({ handleSelect }: ChatSectionProps) => {
  
  const inputRef = useRef<HTMLInputElement>(null);

  const sessionId = useInspectorStore((s) => s.sessionId);
  const conversationList = useInspectorStore((s) => s.conversationList);
  const inputDisabled = useInspectorStore((s) => s.inputDisabled);
  const inspectorMode = useInspectorStore((s) => s.inspectMode);
  const selectedElement = useInspectorStore((s) => s.selectedElement);
  const replayingConversation = useInspectorStore(
    (s) => s.replayingConversation,
  );
  const waitingForResponse = useInspectorStore(selectWaitingForResponse);
  const conversationEventActions = useInspectorStore(
    (s) => s.conversationEventActions,
  );
  const sendMessage = useInspectorStore((s) => s.sendMessage);
  const startNewConversation = useInspectorStore((s) => s.startNewConversation);
  const replayConversationAction = useInspectorStore(
    (s) => s.replayConversation,
  );
  const startVoiceStreaming = useInspectorStore((s) => s.startVoiceStreaming);
  const stopVoiceStreaming = useInspectorStore((s) => s.stopVoiceStreaming);
  const voiceFeaturesEnabled = useInspectorStore((s) => s.voiceFeaturesEnabled);
  const onVoiceErrorRef = useInspectorStore((s) => s.onVoiceErrorRef);
  const flowView = useInspectorStore((s) => s.inspectMode);

  const handleMessageSubmit = useCallback(
    (message: string) => {
      clearSelectedElement();
      sendMessage(message);
    },
    [sendMessage],
  );

  const replayConversationUntilEvent = useCallback(
    (eventId: string) => {
      const conversation = conversationList.find((c) =>
        c.events.some((event) => event?.id === eventId),
      );
      if (!conversation) return;
      const eventsUntil = conversation.events.slice(
        0,
        conversation.events.findIndex((event) => event.id === eventId) + 1,
      );
      replayConversationAction(eventsUntil);
    },
    [conversationList, replayConversationAction],
  );

  const setFlowView = useCallback(
    (value: boolean) => setInspectMode(value),
    [],
  );

  const lightColor = "#FFFFFF";
  const headerSx = {
    borderBottom: "1px solid",
    borderColor: "rasaNeutral.200",
    textAlign: "center",
    px: "1.5rem",
    py: "0.5rem",
  };

  return (
    <ScrollContainer>
      <ScrollFixedHeader css={headerSx}>
        <ChatHeader
          onNewConversation={startNewConversation}
          flowView={flowView}
          setFlowView={setFlowView}
          conversationList={conversationList}
          sessionId={sessionId}
        />
      </ScrollFixedHeader>

      <ScrollContent withSpacing={false} bg={lightColor} mb="1rem">
        <TryAssistantConversation
          replayConversation={replayConversationUntilEvent}
          conversationAssistant={{}}
          conversationList={conversationList}
          onQuickReply={handleMessageSubmit}
          onSelect={handleSelect}
          selectedElementId={selectedElement?.id}
          inspectorMode={inspectorMode}
          waitingForResponse={waitingForResponse}
          replayingConversation={replayingConversation}
          conversationEventActions={conversationEventActions}
        />
      </ScrollContent>

      <ScrollFixedFooter>
        <OnboardingTooltip target="messageInput">
          <Box>
            <MessageInput
              ref={inputRef}
              onSubmit={handleMessageSubmit}
              startVoiceStreaming={startVoiceStreaming}
              stopVoiceStreaming={stopVoiceStreaming}
              onVoiceErrorRef={onVoiceErrorRef}
              isDisabled={inputDisabled}
              voiceFeaturesEnabled={voiceFeaturesEnabled}
            />
          </Box>
        </OnboardingTooltip>
      </ScrollFixedFooter>
    </ScrollContainer>
  );
};
