import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { io, type Socket } from "socket.io-client";
import { v4 as uuid } from "uuid";
import {
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
} from "../types";
import {
  getSlotRelatedEvents,
  isUtterance,
  mapRawEventsToConversationEvents,
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
  stopMicrophoneStream
} from "../utils/voice/audiostream";
import { SocketTimeoutError, SocketUnavailableError } from "../errors";

const REACT_APP_SESSION_HISTORY_KEY = "rasa_session_history";
const SOCKET_IO_RECONNECTION_ATTEMPTS = 3;

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

function isWaitingForUserInput(events: UnionEventType[]): boolean {
  const lastEvent = events[events.length - 1];
  return (
    lastEvent && !isUtterance(lastEvent) && lastEvent.name === "action_listen"
  );
}

function generatePlaceholderId() {
  return uuid().replaceAll("-", "");
}

function generatePlaceholderUserUtterance(message: string): Utterance {
  return {
    id: generatePlaceholderId(),
    text: message,
    tokens: [],
    timestamp: `${Date.now()}`,
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
  onSessionStart,
  onReconnectError,
  onMessageSent,
  useMemoryOnly = true,
}: {
  projectId: string;
  onSessionStart?: (id: string) => void;
  onReconnectError?: (error: unknown) => void;
  onMessageSent?: (message: string) => void;
  useMemoryOnly?: boolean;
}) {
  const { logError, track } = useInspectorContext();
  const [localStorageHistory, setLocalStorageHistory] =
    // TODO: update useLocalStorage to deal with `undefined | null` keys
    // and avoid keys like `rasa_session_history_undefined`;
    // TODO: load projectId from the store instead of the params
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

  const onReconnectErrorRef = useRef(onReconnectError);
  const onSessionStartRef = useRef(onSessionStart);
  const onMessageSentRef = useRef(onMessageSent);
  useEffect(() => {
    onReconnectErrorRef.current = onReconnectError;
  }, [onReconnectError]);
  useEffect(() => {
    onSessionStartRef.current = onSessionStart;
  }, [onSessionStart]);
  useEffect(() => {
    onMessageSentRef.current = onMessageSent;
  }, [onMessageSent]);

  // Select appropriate storage based on parameter
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
  // Track modality for the is_voice parameter in session_request
  const activeModalityRef = useRef<"text" | "voice">("text");
  const [sessionId, setSessionId] = useState(uuid());
  const [url, setUrl] = useState("");
  const [error, setError] = useState<
    ModelServiceError | RasaProError | undefined
  >(undefined);
  const socket = useRef<Socket>(undefined);
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

  const cleanup = () => {
    socket.current?.removeAllListeners();
    socket.current?.io.removeAllListeners();
    socket.current?.disconnect();
  };

  const disableChat = () => {
    setInputDisabled(true);
    setWaitingForUserInput(false);
    setReplayingConversation(false);
  };

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
    //Next line added because linter requires both setConversationHistory and conversationHistory to be in the deps
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversation]);

  useEffect(() => {
    initialTrackerDataRef.current = initialTrackerData;
  }, [initialTrackerData]);

  useEffect(() => {
    if (!socket.current && url) {
      const urlObject = new URL(url);

      const socketIoPath =
        urlObject.pathname === "/" ? undefined : urlObject.pathname;

      socket.current = io(urlObject.toString(), {
        transports: ["websocket", "polling"],
        path: socketIoPath,
        reconnectionAttempts: SOCKET_IO_RECONNECTION_ATTEMPTS,
        reconnectionDelayMax: 2000,
      });

      socket.current?.on("connect", () => {
        socket.current?.emit("session_request", {
          session_id: sessionId,
          is_voice: activeModalityRef.current === "voice",
        });
      });

      socket.current?.on("bot_message", (data) => {
        if (audioQueueRef.current) {
          if (typeof data === "string") {
            try {
              addDataToAudioQueue(audioQueueRef.current)(data);
            } catch (error) {
              logError(error, {
                tags: {
                  component: "useBotConnection",
                  action: "bot_message",
                },
                extra: { data },
              });
            }
          } else {
            logError(
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
          }
        }
      });

      socket.current?.on("session_confirm", () => {
        setInputDisabled(false);
        setError(undefined);
        if (socketReadyPromiseResolveRef.current) {
          socketReadyPromiseResolveRef.current();
          socketReadyPromiseResolveRef.current = undefined;
        }
        if (initialTrackerDataRef.current) {
          setInitialTrackerData(undefined);
          socket.current?.emit("update_tracker", initialTrackerDataRef.current);
          return;
        }

        if (activeModalityRef.current === "text") {
          sendMessage(SESSION_START_MESSAGE);
        }
      });

      socket.current?.on("error", (error) => {
        setError(error as RasaProError);
        disableChat();
      });

      socket.current?.on("disconnect", (reason, details) => {
        if (!socket.current?.active) {
          disableChat();
          logError(reason, {
            tags: {
              component: "useBotConnection",
              action: "disconnect",
            },
            extra: details as { [key: string]: string | null },
          });
        }
      });

      socket.current?.io.on("reconnect_error", (error) => {
        disableChat();
        onReconnectErrorRef?.current?.(error);
        logError(error, {
          tags: {
            component: "useBotConnection",
            action: "reconnect_error",
          },
        });
      });

      socket.current?.io.on("reconnect_failed", () => {
        disableChat();
        onReconnectErrorRef?.current?.(new Error("Reconnect failed"));
        logError(
          `websocket wasn't able to reconnect within ${SOCKET_IO_RECONNECTION_ATTEMPTS} attempts`,
          {
            tags: {
              component: "useBotConnection",
              action: "reconnect_failed",
            },
          },
        );
      });

      socket.current?.on("tracker", (response: TrackerResponseData) => {
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
          setConversation((conv) => {
            return {
              ...conv,
              events,
              totalNumberOfUserMessages: events.filter(
                (event: UnionEventType) => isUtterance(event),
              ).length,
            };
          });
          setReplayingConversation(false);
          setWaitingForUserInput(isWaitingForUserInput(events));
        }
      });

      return () => {
        if (socket.current) {
          cleanup();
        }
      };
    }
  }, [url, sessionId, sendMessage]);

  useEffect(() => {
    onSessionStartRef?.current?.(sessionId);
  }, [sessionId]);

  const reset = () => {
    setUrl("");
    startNewConversation();
  };

  const startNewConversation = useCallback((): string => {
    setInputDisabled(true);
    cleanup();
    socket.current = undefined;
    socketReadyPromiseRef.current = new Promise((resolve) => {
      socketReadyPromiseResolveRef.current = resolve;
    });
    const newSessionId = uuid();
    setConversation(initialConversationState(newSessionId));
    setSessionId(newSessionId);
    setMemoryHistory({});
    return newSessionId;
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
      const audioQueue = createAudioQueue(socket.current);
      audioQueueRef.current = audioQueue;
      await setupAudioPlayback(socket.current, logError, audioQueue);
      microphoneStreamRef.current = await streamMicrophoneToServer(
        socket.current,
        logError,
      );
    } else {
      throw new SocketUnavailableError();
    }
  }, [startNewConversation, logError]);

  const stopVoiceStreaming = useCallback(async () => {
    if (activeModalityRef.current !== "voice") {
      return;
    }
    activeModalityRef.current = "text";
    try {
      await stopMicrophoneStream(microphoneStreamRef.current);
    } catch (error) {
      logError(error, {
        tags: {
          component: "useBotConnection",
          action: "stopMicrophoneStream",
        },
      });
    }
    try {
      await stopAudioPlayback(audioQueueRef.current);
    } catch (error) {
      logError(error, {
        tags: {
          component: "useBotConnection",
          action: "stopAudioPlayback",
        },
      });
    } finally {
      microphoneStreamRef.current = undefined;
      audioQueueRef.current = undefined;
      startNewConversation();
    }
  }, [startNewConversation]);

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

  return {
    conversationList,
    stack,
    inputDisabled,
    sessionId,
    slots,
    slotRelatedEvents,
    replayingConversation,
    waitingForUserInput,
    error,
    setUrl,
    reset,
    sendMessage,
    replayConversation,
    setInputDisabled,
    startNewConversation,
    startVoiceStreaming,
    stopVoiceStreaming,
    cleanup,
  };
}
