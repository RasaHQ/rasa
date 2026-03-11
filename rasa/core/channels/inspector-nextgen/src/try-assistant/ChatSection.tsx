import { Box } from "@chakra-ui/react";
import { type RefObject, useRef } from "react";
import { OnboardingTooltip } from "../OnboardingTooltip";
import { type Conversation, type ConversationEventAction, type UnionEventType, type VoiceErrorHandler } from "../types";
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
  sessionId: string;
  replayConversation?: (eventId: string) => void;
  onNewConversation: () => void;
  conversationAssistants: Record<string, string>;
  conversationList: Conversation[];
  handleMessageSubmit: (payload: string) => void;
  handleSelect: (selection: UnionEventType) => void;
  selectedElement: UnionEventType | undefined;
  inputDisabled: boolean;
  inspectorMode: boolean;
  waitingForResponse: boolean;
  replayingConversation: boolean;
  conversationEventActions?: ConversationEventAction[];
  setFlowView: (flowView: boolean) => void;
  flowView: boolean;
  startVoiceStreaming: () => Promise<void>;
  stopVoiceStreaming: () => Promise<void>;
  onVoiceErrorRef: RefObject<VoiceErrorHandler>;
  voiceFeaturesEnabled: boolean;
}

export const ChatSection = ({
  sessionId,
  replayConversation,
  conversationAssistants,
  conversationList,
  handleMessageSubmit,
  handleSelect,
  selectedElement,
  inputDisabled,
  inspectorMode,
  waitingForResponse,
  replayingConversation,
  onNewConversation,
  conversationEventActions,
  setFlowView,
  flowView,
  startVoiceStreaming,
  stopVoiceStreaming,
  onVoiceErrorRef,
  voiceFeaturesEnabled,
}: ChatSectionProps) => {
  const inputRef = useRef<HTMLInputElement>(null);

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
          onNewConversation={onNewConversation}
          flowView={flowView}
          setFlowView={setFlowView}
          conversationList={conversationList}
          sessionId={sessionId}
        />
      </ScrollFixedHeader>

      <ScrollContent withSpacing={false} bg={lightColor} mb="1rem">
        <TryAssistantConversation
          replayConversation={replayConversation}
          conversationAssistant={conversationAssistants}
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
