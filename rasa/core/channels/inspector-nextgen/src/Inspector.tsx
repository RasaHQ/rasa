import { useEffect, useState } from "react";
import { v4 as uuid } from "uuid";
import { useBotConnection } from "./hooks/useBotConnection";
import { InspectorContextProvider } from "./InspectorContext";
import { TryAssistant } from "./try-assistant/TryAssistant";
import type {
  ConversationEventAction,
  LogErrorFn,
  OnboardingTooltipConfig,
  TrackFn,
} from "./types";

type Props = {
  projectUrl: string;
  botDataEndpoint: string;
  conversationEventActions?: ConversationEventAction[];
  projectId?: string;
  singleSessionMode?: boolean;
  initialInspectMode?: boolean;
  onSessionStart?: (sessionId: string) => void;
  onInspectModeChange?: (inspectMode: boolean) => void;
  onMessageSent?: (message: string) => void;
  onReconnectError?: (error: unknown) => void;
  track?: TrackFn;
  logError?: LogErrorFn;
  onboardingTooltips?: OnboardingTooltipConfig[];
  voiceFeaturesEnabled?: boolean;
};

export const Inspector = ({ track, logError, onboardingTooltips, ...rest }: Props) => (
  <InspectorContextProvider track={track} logError={logError} onboardingTooltips={onboardingTooltips}>
    <InspectorContent {...rest} />
  </InspectorContextProvider>
);

const InspectorContent = (props: Omit<Props, "track" | "logError" | "onboardingTooltips">) => {
  const {
    projectUrl,
    botDataEndpoint,
    conversationEventActions,
    singleSessionMode,
    projectId,
    initialInspectMode = false,
    onSessionStart,
    onReconnectError,
    onInspectModeChange,
    onMessageSent,
    voiceFeaturesEnabled = true,
  } = props;
  const [inspectMode, setInspectMode] = useState<boolean>(initialInspectMode);
  const {
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
    startVoiceStreaming,
    stopVoiceStreaming,
  } = useBotConnection({
    projectId: projectId ?? uuid(),
    useMemoryOnly: singleSessionMode ?? false,
    onSessionStart,
    onReconnectError,
    onMessageSent,
  });

  useEffect(() => {
    onInspectModeChange?.(inspectMode);
  }, [inspectMode, onInspectModeChange]);

  return (
    <TryAssistant
      projectUrl={projectUrl}
      flowView={inspectMode}
      setFlowView={setInspectMode}
      sendMessage={sendMessage}
      setUrl={setUrl}
      startNewConversation={startNewConversation}
      conversationList={conversationList}
      inputDisabled={inputDisabled}
      sessionId={sessionId}
      stack={stack}
      replayingConversation={replayingConversation}
      waitingForUserInput={waitingForUserInput}
      replayConversation={replayConversation}
      conversationEventActions={conversationEventActions}
      botDataEndpoint={botDataEndpoint}
      startVoiceStreaming={startVoiceStreaming}
      stopVoiceStreaming={stopVoiceStreaming}
      voiceFeaturesEnabled={voiceFeaturesEnabled}
    />
  );
}
