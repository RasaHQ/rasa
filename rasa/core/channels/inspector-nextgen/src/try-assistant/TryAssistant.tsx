import { Box, Flex } from "@chakra-ui/react";
import { type RefObject, useEffect, useMemo, useState } from "react";
import { useConversationData } from "../hooks/useConversationData";
import {
  type Conversation,
  type ConversationEventAction,
  type Stack,
  type UnionEventType,
  type VoiceErrorHandler,
} from "../types";
import { ChatSection } from "./ChatSection";
import { EventDetails } from "./EventDetails";
import { FlowSection } from "./FlowSection";

type Props = {
  flowView: boolean;
  sendMessage: (message: string) => void;
  setUrl: (url: string) => void;
  startNewConversation: () => void;
  conversationList: Conversation[];
  inputDisabled: boolean;
  sessionId: string;
  stack: Stack[];
  replayingConversation: boolean;
  waitingForUserInput: boolean;
  replayConversation: (events: UnionEventType[]) => void;
  projectUrl: string;
  conversationEventActions?: ConversationEventAction[];
  setFlowView: (flowView: boolean) => void;
  botDataEndpoint: string;
  startVoiceStreaming: () => Promise<void>;
  stopVoiceStreaming: () => Promise<void>;
  onVoiceErrorRef: RefObject<VoiceErrorHandler>;
  voiceFeaturesEnabled: boolean;
};

export function TryAssistant({
  flowView,
  sendMessage,
  setUrl,
  startNewConversation,
  conversationList,
  inputDisabled,
  sessionId,
  stack,
  replayingConversation,
  waitingForUserInput,
  replayConversation,
  projectUrl,
  setFlowView,
  conversationEventActions,
  botDataEndpoint,
  startVoiceStreaming,
  stopVoiceStreaming,
  onVoiceErrorRef,
  voiceFeaturesEnabled,
}: Readonly<Props>) {
  const { flows, isLoading: flowsLoading, error: flowsError } = useConversationData(projectUrl, botDataEndpoint);

  const allowedPatterns = ["pattern_session_start", "pattern_completed"];

  const isUserVisibleFrame = (frame: Stack) => {
    return (
      !frame.flowId?.startsWith("pattern_") ||
      allowedPatterns.includes(frame.flowId)
    );
  };

  const latestStack = stack.slice().reverse().find(isUserVisibleFrame);

  const [selectedElement, setSelectedElement] = useState<UnionEventType>();
  const conversationForSelectedElement = selectedElement?.id
    ? conversationList.find((conversation) =>
      conversation.events.some((event) => event?.id === selectedElement.id),
    )
    : undefined;

  const stackToShow: Stack | undefined = useMemo(() => {
    if (
      selectedElement?.metadata?.flow_id &&
      selectedElement?.metadata?.step_id
    ) {
      return {
        frameId: selectedElement.id,
        flowId: selectedElement.metadata.flow_id,
        stepId: selectedElement.metadata.step_id,
        ended: false,
      };
    }
    return latestStack;
  }, [selectedElement, latestStack]);

  const handleMessageSubmit = (message: string) => {
    setSelectedElement(undefined);
    sendMessage(message);
  };

  const handleSelect = (selection: UnionEventType) => {
    if (selection.id === selectedElement?.id) {
      setSelectedElement(undefined);
    } else {
      setSelectedElement(selection);
    }
  };

  const replayConversationUntilSelectedElement = (eventId: string) => {
    const conversation = conversationList.find((conversation) =>
      conversation.events.some((event) => event?.id === eventId),
    );
    if (!conversation) {
      return;
    }
    const eventsUntilSelectedElement = conversation?.events.slice(
      0,
      conversation.events.findIndex((event) => event.id === eventId) + 1,
    );
    replayConversation(eventsUntilSelectedElement);
  };

  const separatorColor = "rasaNeutral.400";
  const canvasSx = {
    borderLeft: "1px solid",
    borderColor: separatorColor,
  };

  useEffect(() => {
    if (projectUrl) {
      setUrl(projectUrl);
    }
  }, [projectUrl, setUrl]);

  return (
    <Flex flexGrow="1" data-testid="try-assistant-container">
      <Box flexBasis={flowView ? "50%" : "100%"}>
        <ChatSection
          inspectorMode={flowView}
          sessionId={sessionId}
          replayConversation={replayConversationUntilSelectedElement}
          conversationAssistants={{}}
          conversationList={conversationList}
          handleMessageSubmit={handleMessageSubmit}
          handleSelect={handleSelect}
          selectedElement={selectedElement}
          inputDisabled={inputDisabled}
          waitingForResponse={!waitingForUserInput && !inputDisabled}
          replayingConversation={replayingConversation}
          onNewConversation={startNewConversation}
          conversationEventActions={conversationEventActions}
          setFlowView={setFlowView}
          flowView={flowView}
          startVoiceStreaming={startVoiceStreaming}
          stopVoiceStreaming={stopVoiceStreaming}
          onVoiceErrorRef={onVoiceErrorRef}
          voiceFeaturesEnabled={voiceFeaturesEnabled}
        />
      </Box>
      {flowView && (
        <Box
          css={canvasSx}
          flexBasis="calc(50% + 1rem)"
          data-testid="inspector-canvas"
        >
          {selectedElement ? (
            <EventDetails
              event={selectedElement}
              onClose={() => setSelectedElement(undefined)}
              flows={flows}
            />
          ) : (
            <FlowSection
              stackToShow={stackToShow}
              conversationList={conversationList}
              conversationForSelectedElement={conversationForSelectedElement}
              flows={flows}
              flowsLoading={flowsLoading}
              flowsError={flowsError}
            />
          )}
        </Box>
      )}
    </Flex>
  );
}
