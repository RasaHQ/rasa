import { useEffect, useRef } from "react";
import { v4 as uuid } from "uuid";
import { useBotConnection } from "./hooks/useBotConnection";
import { InspectorContextProvider } from "./InspectorContext";
import {
  initInspectorStore,
  inspectorStore,
  type InspectorStoreState,
} from "./store";
import { Toaster } from "./Toaster";
import { TryAssistant } from "./try-assistant/TryAssistant";
import type {
  ConversationEventAction,
  LogErrorFn,
  OnboardingTooltipConfig,
  ShowToastFn,
  TrackFn,
} from "./types";

type Props = {
  projectUrl: string;
  botDataEndpoint: string;
  conversationEventActions?: ConversationEventAction[];
  projectId?: string;
  singleSessionMode?: boolean;
  initialInspectMode?: boolean;
  isEmbedded?: boolean;
  onSessionStart?: (sessionId: string) => void;
  onInspectModeChange?: (inspectMode: boolean) => void;
  onMessageSent?: (message: string) => void;
  onReconnectError?: (error: unknown) => void;
  track?: TrackFn;
  logError?: LogErrorFn;
  showToast?: ShowToastFn;
  onboardingTooltips?: OnboardingTooltipConfig[];
  voiceFeaturesEnabled?: boolean;
  socketReconnectAttempts?: number;
  currentSessionId?: number;
  sessionId?: string;
  resetSession?: () => void;
  trackerEndpoint?: string;
};

export const Inspector = ({
  track,
  logError,
  showToast,
  onboardingTooltips,
  socketReconnectAttempts,
  ...rest
}: Props) => (
  <InspectorContextProvider
    track={track}
    logError={logError}
    showToast={showToast}
    onboardingTooltips={onboardingTooltips}
    socketReconnectAttempts={socketReconnectAttempts}
  >
    <InspectorContent {...rest} />
    {!showToast && <Toaster />}
  </InspectorContextProvider>
);

const InspectorContent = (
  props: Omit<Props, "track" | "logError" | "onboardingTooltips">,
) => {
  const {
    projectUrl,
    singleSessionMode,
    projectId,
    sessionId,
    resetSession,
    onSessionStart,
    onReconnectError,
    onMessageSent,
  } = props;

  const initialized = useRef<boolean>(null);
  if (initialized.current === null) {
    initialized.current = true;
    initInspectorStore({
      projectUrl,
      botDataEndpoint: props.botDataEndpoint,
      trackerEndpoint: props.trackerEndpoint,
      conversationEventActions: props.conversationEventActions ?? [],
      voiceFeaturesEnabled: props.voiceFeaturesEnabled ?? true,
      inspectMode: props.initialInspectMode ?? false,
      isEmbedded: props.isEmbedded ?? false,
    });
  }

  useBotConnection({
    projectId: projectId ?? uuid(),
    useMemoryOnly: singleSessionMode ?? false,
    sessionId,
    resetSession,
    onSessionStart,
    onReconnectError,
    onMessageSent,
  });

  useEffect(() => {
    const updates: Partial<InspectorStoreState> = {};

    if (projectUrl) updates.projectUrl = projectUrl;
    if (props.botDataEndpoint) updates.botDataEndpoint = props.botDataEndpoint;
    if (props.conversationEventActions)
      updates.conversationEventActions = props.conversationEventActions;
    if (typeof props.voiceFeaturesEnabled === "boolean")
      updates.voiceFeaturesEnabled = props.voiceFeaturesEnabled;

    if (Object.keys(updates).length > 0) {
      inspectorStore.setState((prev) => ({ ...prev, ...updates }));
    }
  }, [
    projectUrl,
    props.botDataEndpoint,
    props.conversationEventActions,
    props.voiceFeaturesEnabled,
  ]);

  return <TryAssistant onInspectModeChange={props.onInspectModeChange} />;
};
