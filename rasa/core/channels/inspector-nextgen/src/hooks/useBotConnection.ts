import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { io, type Socket } from "socket.io-client";
import { v4 as uuid } from "uuid";
import {
  ConversationEventType,
  UtteranceType,
  type Conversation,
  type ConversationEvent,
  type ModelServiceError,
  type RasaProError,
  type RawEvent,
  type RawStack,
  type SlotState,
  type Stack,
  type TrackerResponseData,
  type UnionEventType,
  type Utterance,
  type VoiceLatency,
} from "../types";
import {
  getSlotRelatedEvents,
  isUtterance,
  isVoiceLatency,
  mapRawEventsToConversationEvents,
  preserveVoiceLatencyOnBotUtterances,
} from "../utils";
import { useInspectorContext } from "../InspectorContext";
import { useLocalStorage } from "./useLocalStorage";
import {
  streamMicrophoneToServer,
  addDataToAudioQueue,
  type AudioQueue,
  createAudioQueue,
  setupAudioPlayback,
  stopAudioPlayback,
  stopMicrophoneStream,
} from "../utils/voice/audiostream";
import { SocketTimeoutError, SocketUnavailableError } from "../errors";
import { inspectorStore, useInspectorStore } from "../store";
import { getConversationHistory } from "../api";

const REACT_APP_SESSION_HISTORY_KEY = "rasa_session_history";

type ConversationHistory = Record<string, Conversation>;

const SESSION_START_MESSAGE = "/session_start";

function formatSlots(slots: { [key: string]: unknown }): SlotState[] {
  if (!slots) {
    return [];
  }

  return Object.entries(slots)
    .filter((slotDuple) => slotDuple[1] != null)
    .map((slotDuple) => ({ name: slotDuple[0], value: slotDuple[1] }));
}

const AGENT_INACTIVE_EVENTS = new Set([
  ConversationEventType.AgentCompleted,
  ConversationEventType.AgentCancelled,
  ConversationEventType.AgentInterrupted,
]);

const AGENT_ACTIVE_EVENTS = new Set([
  ConversationEventType.AgentStarted,
  ConversationEventType.AgentResumed,
]);

function hasActiveSubAgent(events: UnionEventType[]): boolean {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (isUtterance(event)) continue;
    if (AGENT_INACTIVE_EVENTS.has(event.conversationEventType)) return false;
    if (AGENT_ACTIVE_EVENTS.has(event.conversationEventType)) return true;
  }
  return false;
}

function isWaitingForUserInput(events: UnionEventType[]): boolean {
  let userMessageAfterAction = false;

  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (isUtterance(event)) {
      if (event.type === UtteranceType.User) userMessageAfterAction = true;
      continue;
    }
    if (event.conversationEventType !== ConversationEventType.Action) continue;
    if (userMessageAfterAction) return false;
    if (event.name === "action_listen") return !hasActiveSubAgent(events);
    return event.name === "action_agent_request_user_input";
  }
  return false;
}

function generatePlaceholderId() {
  return uuid().replaceAll("-", "");
}

function generatePlaceholderUserUtterance(message: string): Utterance {
  const timestamp = new Date();
  return {
    id: generatePlaceholderId(),
    text: message,
    tokens: [],
    timestamp: `${timestamp.toISOString()}`,
    originalTimestamp: +timestamp,
    type: UtteranceType.User,
    __typename: "Utterance",
    rephrase: false,
    rephrasePrompt: null,
    metadata: {
      parseData: {},
    },
  };
}

export function useBotConnection({
  projectId,
  useMemoryOnly = true,
  sessionId: externalSessionId,
  resetSession: resetExternalSession,
  onSessionStart,
  onReconnectError,
  onMessageSent,
}: {
  projectId: string;
  useMemoryOnly?: boolean;
  sessionId?: string;
  resetSession?: () => void;
  onSessionStart?: (id: string) => void;
  onReconnectError?: (error: unknown) => void;
  onMessageSent?: (message: string) => void;
}) {
  const { logError, showToast, socketReconnectAttempts, track } =
    useInspectorContext();
  const [localStorageHistory, setLocalStorageHistory] =
    useLocalStorage<ConversationHistory>(
      `${REACT_APP_SESSION_HISTORY_KEY}_${projectId}`,
      {},
    );
  const [memoryHistory, setMemoryHistory] = useState<ConversationHistory>({});

  // Socket readiness coordination using deferred promise pattern.
  // We need to wait for the backend's session_confirm event before setting up audio streams,
  // otherwise the microphone starts emitting data to an undefined socket (race condition).
  //
  // How it works:
  // 1. startNewConversation() creates a new Promise and stores the resolve function
  // 2. Backend sends session_confirm → we call the resolve function
  // 3. startVoiceStreaming() awaits the Promise before initializing audio
  const socketReadyPromiseRef = useRef<Promise<void> | undefined>(undefined);
  const socketReadyPromiseResolveRef = useRef<
    ((value: void) => void) | undefined
  >(undefined);

  const logErrorRef = useRef(logError);
  const onReconnectErrorRef = useRef(onReconnectError);
  const onSessionStartRef = useRef(onSessionStart);
  const onMessageSentRef = useRef(onMessageSent);
  const resetExternalSessionRef = useRef(resetExternalSession);
  useEffect(() => {
    logErrorRef.current = logError;
  }, [logError]);
  useEffect(() => {
    onReconnectErrorRef.current = onReconnectError;
  }, [onReconnectError]);
  useEffect(() => {
    onSessionStartRef.current = onSessionStart;
  }, [onSessionStart]);
  useEffect(() => {
    onMessageSentRef.current = onMessageSent;
  }, [onMessageSent]);
  useEffect(() => {
    resetExternalSessionRef.current = resetExternalSession;
  }, [resetExternalSession]);

  const [conversationHistory, setConversationHistory] = useMemoryOnly
    ? [memoryHistory, setMemoryHistory]
    : [localStorageHistory, setLocalStorageHistory];
  const initialConversationState = (conversationId: string) => ({
    events: [],
    id: conversationId,
    startDate: new Date().toISOString(),
    tags: [],
    reviewed: false,
    totalNumberOfUserMessages: 0,
  });
  const activeModalityRef = useRef<"text" | "voice">("text");
  const [sessionId, setSessionId] = useState(externalSessionId || uuid());
  const [url, setUrl] = useState("");
  const [, setError] = useState<ModelServiceError | RasaProError | undefined>(
    undefined,
  );
  const onVoiceErrorRef = useRef<((err: RasaProError) => void) | null>(null);
  const { projectUrl, trackerEndpoint } = useInspectorStore((s) => ({ projectUrl: s.projectUrl, trackerEndpoint: s.trackerEndpoint }));

  useEffect(() => {
    if (projectUrl) {
      setUrl(projectUrl);
    }
  }, [projectUrl]);

  const socket = useRef<Socket>(undefined);
  const sampleRateRef = useRef<number>(48000);
  const audioQueueRef = useRef<AudioQueue>(undefined);
  const microphoneStreamRef =
    useRef<
      ReturnType<typeof streamMicrophoneToServer> extends Promise<infer T>
      ? T
      : never
    >(undefined);
  const [conversation, setConversation] = useState<Conversation>(
    initialConversationState(sessionId),
  );
  const [stack, setStack] = useState<Stack[]>([]);
  const [initialTrackerData, setInitialTrackerData] = useState<
    | {
      sender_id: string;
      events: (RawEvent | undefined)[];
    }
    | undefined
  >(undefined);
  const [inputDisabled, setInputDisabled] = useState(true);
  const [slots, setSlots] = useState<SlotState[]>([]);
  const initialTrackerDataRef = useRef(initialTrackerData);
  const [slotRelatedEvents, setSlotRelatedEvents] = useState<
    ConversationEvent[]
  >([]);
  const [waitingForUserInput, setWaitingForUserInput] = useState(false);
  const [replayingConversation, setReplayingConversation] = useState(false);
  const [voiceLatency, setVoiceLatency] = useState<VoiceLatency | undefined>();

  const cleanup = () => {
    setVoiceLatency(undefined);
    socket.current?.removeAllListeners();
    socket.current?.io.removeAllListeners();
    socket.current?.disconnect();
  };

  const disableChat = useCallback(() => {
    setInputDisabled(true);
    setWaitingForUserInput(false);
    setReplayingConversation(false);
  }, []);

  const sendMessage = useCallback(
    (message: string) => {
      socket.current?.emit("user_message", { message, session_id: sessionId });
      if (message === SESSION_START_MESSAGE) {
        return;
      }

      onMessageSentRef.current?.(message);
      void track("User Message Sent", { target: "rasa_agent" });

      const userUtterance: Utterance =
        generatePlaceholderUserUtterance(message);
      setConversation((conv) => ({
        ...conv,
        events: [...conv.events, userUtterance],
      }));
      setWaitingForUserInput(false);
      setReplayingConversation(false);
    },
    [sessionId, track],
  );

  useEffect(() => {
    setConversationHistory({
      ...conversationHistory,
      [conversation.id]: conversation,
    });
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversation]);

  useEffect(() => {
    initialTrackerDataRef.current = initialTrackerData;
  }, [initialTrackerData]);

  useEffect(() => {
    if (!voiceLatency) return;

    setConversation((currentConversations) => {
      const events = currentConversations.events;
      let lastBotIndex = -1;
      for (let i = events.length - 1; i >= 0; i--) {
        const event = events[i];
        if (isUtterance(event) && event.type === UtteranceType.Bot) {
          lastBotIndex = i;
          break;
        }
      }
      if (lastBotIndex < 0) return currentConversations;
      const bot = events[lastBotIndex];
      if (!isUtterance(bot) || bot.metadata?.voiceLatency) return currentConversations;
      const next = [...events];
      next[lastBotIndex] = {
        ...bot,
        metadata: { ...bot.metadata, voiceLatency },
      };
      return { ...currentConversations, events: next };
    });
  }, [voiceLatency]);

  useEffect(() => {
    if (!socket.current && url) {
      const urlObject = new URL(url);

      const socketIoPath =
        urlObject.pathname === "/" ? undefined : `${urlObject.pathname}/socket.io`;

      socket.current = io(urlObject.toString(), {
        transports: ["websocket", "polling"],
        path: socketIoPath,
        reconnectionAttempts: socketReconnectAttempts,
        reconnectionDelayMax: 2000,
      });

      socket.current?.on("connect", () => {
        socket.current?.emit("session_request", {
          session_id: sessionId,
          is_voice: activeModalityRef.current === "voice",
        });
      });

      const storeVoiceLatencyFromBotMessage = (payload: string) => {
        try {
          const parsed = JSON.parse(payload) as { latency?: unknown };
          if (parsed.latency !== undefined && isVoiceLatency(parsed.latency)) {
            setVoiceLatency(parsed.latency);
          }
        } catch {
          // Not JSON (e.g. raw audio envelope) — ignore
        }
      };

      socket.current?.on("bot_message", (data) => {
        if (activeModalityRef.current === "text") {
          return;
        }
        if (typeof data !== "string") {
          logErrorRef.current(
            `Unexpected typeof bot_message data. Got: ${typeof data}, expected: string.`,
            {
              tags: {
                component: "useBotConnection",
                action: "bot_message",
              },
              extra: {
                data: typeof data === "object" ? JSON.stringify(data) : null,
              },
            },
          );
          return;
        }
        storeVoiceLatencyFromBotMessage(data);
        if (audioQueueRef.current) {
          try {
            addDataToAudioQueue(audioQueueRef.current)(data);
          } catch (error) {
            logErrorRef.current(error, {
              tags: {
                component: "useBotConnection",
                action: "bot_message",
              },
              extra: { data },
            });
          }
        }
      });

      socket.current?.on(
        "session_confirm",
        (data: string | { session_id: string; sample_rate: number }) => {
          if (typeof data === "object" && data !== null && "sample_rate" in data) {
            sampleRateRef.current = data.sample_rate;
          }
          setInputDisabled(false);
          setError(undefined);
          if (socketReadyPromiseResolveRef.current) {
            socketReadyPromiseResolveRef.current();
            socketReadyPromiseResolveRef.current = undefined;
          }
          if (initialTrackerDataRef.current) {
            setInitialTrackerData(undefined);
            socket.current?.emit(
              "update_tracker",
              initialTrackerDataRef.current,
            );
            return;
          }

          if (activeModalityRef.current === "text") {
            if (!externalSessionId) {
              onSessionStartRef?.current?.(sessionId);
              sendMessage(SESSION_START_MESSAGE);
              // quick fix for new sessions after reconnecting, needs more attention in the future
              setConversation({
                ...conversation,
                startDate: new Date().toISOString(),
              });
            } else if (externalSessionId && trackerEndpoint) {
              getConversationHistory({ projectUrl, trackerEndpoint })
                .then((trackerResult) =>
                  handleTrackerResponse(trackerResult)
                )
                .catch((err) => {
                  logError(err);
                });
            }
          }
        },
      );

      socket.current?.on("error", (error) => {
        setError(error as RasaProError);
        showToast({
          title: "An internal error has happened",
          description: `${error}`,
          type: "error",
        });
        disableChat();
      });

      socket.current?.on("voice_error", (error) => {
        logErrorRef.current(error, {
          tags: {
            component: "useBotConnection",
            action: "voice_error",
          },
        });
        onVoiceErrorRef.current?.(error as RasaProError);
      });

      socket.current?.on("disconnect", (reason, details) => {
        if (activeModalityRef.current === "voice") {
          onVoiceErrorRef.current?.({
            error: "connection_lost",
            message: "Server connection lost during voice call",
          });
        } else {
          showToast({
            title: "Server disconnected",
            description: "Trying to reconnect...",
            type: "error",
          });
        }
        if (!socket.current?.active) {
          disableChat();
          logErrorRef.current(reason, {
            tags: {
              component: "useBotConnection",
              action: "disconnect",
            },
            extra: details as { [key: string]: string | null },
          });
        }
      });

      socket.current?.io.on("reconnect_error", (error) => {
        showToast({
          title: "Reconnect failed",
          description: "Reconnecting...",
          type: "error",
        });
        disableChat();
        onReconnectErrorRef?.current?.(error);
        logErrorRef.current(error, {
          tags: {
            component: "useBotConnection",
            action: "reconnect_error",
          },
        });
      });

      socket.current?.io.on("reconnect_failed", () => {
        if (activeModalityRef.current === "voice") {
          onVoiceErrorRef.current?.({
            error: "connection_lost",
            message: "Server connection lost during voice call",
          });
        }
        const errorDescription = `websocket wasn't able to reconnect ${socketReconnectAttempts ? `within ${socketReconnectAttempts} attempts` : ""}`;
        showToast({
          title: "Reconnect failed",
          description: errorDescription,
          type: "error",
        });
        disableChat();
        onReconnectErrorRef?.current?.(new Error("Reconnect failed"));
        logError(errorDescription, {
          tags: {
            component: "useBotConnection",
            action: "reconnect_failed",
          },
        });
      });

      socket.current?.on("tracker", handleTrackerResponse);

      return () => {
        if (socket.current) {
          cleanup();
        }
      };
    }
    /* eslint-disable-next-line react-hooks/exhaustive-deps */
  }, [
    url,
    sessionId,
    sendMessage,
    logError,
    showToast,
    socketReconnectAttempts,
  ]);

  const handleTrackerResponse = useCallback((response: TrackerResponseData) => {
    if (!response) return;

    const shouldProcessResponse = response.sender_id === sessionId;
    if (shouldProcessResponse) {
      const events = mapRawEventsToConversationEvents(response.events);
      setSlotRelatedEvents(getSlotRelatedEvents(events));

      setSlots(formatSlots(response.slots));
      const convertedStack: Stack[] = response.stack.map(
        (item: RawStack) => ({
          frameId: item.frame_id,
          flowId: item.flow_id,
          stepId: item.step_id,
          collect: item.collect,
          utter: item.utter,
          ended: false,
        }),
      );
      if (convertedStack.length > 0) {
        setStack(convertedStack);
      }
      const startDate = events?.[0]?.timestamp;
      setConversation((conv) => {
        const newEvents =
          activeModalityRef.current === "voice"
            ? preserveVoiceLatencyOnBotUtterances(conv.events, events)
            : events;
        return {
          ...conv,
          events: newEvents,
          totalNumberOfUserMessages: newEvents.filter(
            (event: UnionEventType) => isUtterance(event),
          ).length,
          startDate: startDate ? new Date(startDate).toISOString() : conv.startDate,
        };
      });
      setReplayingConversation(false);
      setWaitingForUserInput(isWaitingForUserInput(events));
    }
  }, [sessionId]);

  const startNewConversation = useCallback((): string => {
    setInputDisabled(true);
    cleanup();
    socket.current = undefined;
    socketReadyPromiseRef.current = new Promise((resolve) => {
      socketReadyPromiseResolveRef.current = resolve;
    });
    resetExternalSessionRef?.current?.();
    const newSessionId = uuid();
    setConversation(initialConversationState(newSessionId));
    setSessionId(newSessionId);
    setMemoryHistory({});
    setStack([]);
    setSlots([]);
    setSlotRelatedEvents([]);
    return newSessionId;
    // IMPORTANT: adding deps to the array might break inspector
  }, []);

  const startVoiceStreaming = useCallback(async () => {
    activeModalityRef.current = "voice";
    startNewConversation();

    if (socketReadyPromiseRef.current) {
      let timeoutId: NodeJS.Timeout;

      const timeoutPromise = new Promise<never>((_resolve, reject) => {
        timeoutId = setTimeout(() => {
          socketReadyPromiseResolveRef.current = undefined;
          socketReadyPromiseRef.current = undefined;
          reject(new SocketTimeoutError());
        }, 10000);
      });

      try {
        await Promise.race([socketReadyPromiseRef.current, timeoutPromise]);
      } finally {
        clearTimeout(timeoutId!);
      }
    }

    if (socket.current) {
      const sr = sampleRateRef.current;
      const audioQueue = createAudioQueue(socket.current);
      audioQueueRef.current = audioQueue;
      await setupAudioPlayback(socket.current, logErrorRef.current, sr, audioQueue);
      microphoneStreamRef.current = await streamMicrophoneToServer(
        socket.current,
        logErrorRef.current,
        sr,
      );
    } else {
      throw new SocketUnavailableError();
    }
    // IMPORTANT: adding deps to the array might break inspector
  }, [startNewConversation]);

  const stopVoiceStreaming = useCallback(async () => {
    if (activeModalityRef.current !== "voice") {
      return;
    }
    activeModalityRef.current = "text";
    try {
      await stopMicrophoneStream(microphoneStreamRef.current);
    } catch (error) {
      logErrorRef.current(error, {
        tags: {
          component: "useBotConnection",
          action: "stopMicrophoneStream",
        },
      });
    }
    try {
      await stopAudioPlayback(audioQueueRef.current);
    } catch (error) {
      logErrorRef.current(error, {
        tags: {
          component: "useBotConnection",
          action: "stopAudioPlayback",
        },
      });
    } finally {
      microphoneStreamRef.current = undefined;
      audioQueueRef.current = undefined;
      setConversation((conv) => ({
        ...conv,
        endDate: (new Date()).toISOString(),
      }));
      disableChat();
    }
    // IMPORTANT: adding deps to the array might break inspector
  }, [disableChat]);

  const replayConversation = useCallback(
    (events: UnionEventType[]) => {
      setReplayingConversation(true);
      const newSessionId = startNewConversation();
      setInitialTrackerData({
        sender_id: newSessionId,
        events: events.map((event) => event.metadata.rawEvent),
      });
    },
    [setReplayingConversation, startNewConversation],
  );

  const conversationList = useMemo(
    () =>
      Object.values(conversationHistory).sort((a, b) => {
        return (
          new Date(a.startDate).getTime() - new Date(b.startDate).getTime()
        );
      }),
    [conversationHistory],
  );

  // --- Sync state and actions to the store ---

  const setUrlAction = useCallback((newUrl: string) => setUrl(newUrl), []);

  // Initial sync via layout effect: runs synchronously after render but
  // before the browser paints, so children see real values on first paint.
  const initialSyncDone = useRef(false);
  useLayoutEffect(() => {
    if (initialSyncDone.current) return;
    initialSyncDone.current = true;
    inspectorStore.setState((prev) => ({
      ...prev,
      sessionId,
      conversationList,
      stack,
      inputDisabled,
      replayingConversation,
      waitingForUserInput,
      slots,
      slotRelatedEvents,
      sendMessage,
      startNewConversation,
      replayConversation,
      setUrl: setUrlAction,
      startVoiceStreaming,
      stopVoiceStreaming,
      onVoiceErrorRef,
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ongoing sync: pushes updates whenever local state or actions change.
  useEffect(() => {
    inspectorStore.setState((prev) => ({
      ...prev,
      sessionId,
      conversationList,
      stack,
      inputDisabled,
      replayingConversation,
      waitingForUserInput,
      slots,
      slotRelatedEvents,
      sendMessage,
      startNewConversation,
      replayConversation,
      setUrl: setUrlAction,
      startVoiceStreaming,
      stopVoiceStreaming,
      onVoiceErrorRef,
    }));
  }, [
    sessionId,
    conversationList,
    stack,
    inputDisabled,
    replayingConversation,
    waitingForUserInput,
    onVoiceErrorRef,
    setUrl,
    slots,
    slotRelatedEvents,
    sendMessage,
    startNewConversation,
    replayConversation,
    setUrlAction,
    startVoiceStreaming,
    stopVoiceStreaming,
  ]);
}
