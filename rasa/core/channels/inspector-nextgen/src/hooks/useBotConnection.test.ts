import { renderHook, act } from "@testing-library/react";
import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
  afterEach,
  type MockedFunction,
} from "vitest";
import { useBotConnection } from "./useBotConnection";
import { useParams } from "react-router-dom";
import { useLocalStorage } from "../hooks/useLocalStorage";
import { REACT_APP_SESSION_HISTORY_KEY } from "../constants";
import { SocketTimeoutError } from "../errors";
import { initInspectorStore, inspectorStore } from "../store";

vi.mock("react-router-dom", () => ({
  useParams: vi.fn(),
}));

vi.mock("../hooks/useLocalStorage", () => ({
  useLocalStorage: vi.fn(),
}));

const mockCreateAudioQueue = vi.fn(() => ({
  marks: [],
  queuedSamples: 0,
  socket: {},
  enqueue: vi.fn(),
  onSamplesPlayed: vi.fn(),
  attachPlaybackNode: vi.fn(),
  addMarker: vi.fn(),
  reduceMarkers: vi.fn(),
  popMarkers: vi.fn(),
  clear: vi.fn(),
}));
const mockSetupAudioPlayback = vi.fn().mockResolvedValue(undefined);
const mockStopAudioPlayback = vi.fn().mockResolvedValue(undefined);
const mockStreamMicrophoneToServer = vi.fn().mockResolvedValue(undefined);
const mockStopMicrophoneStream = vi.fn().mockResolvedValue(undefined);
const mockAddDataToAudioQueue = vi.fn(() => vi.fn());

vi.mock("../utils/voice/audiostream", () => ({
  createAudioQueue: () => mockCreateAudioQueue(),
  setupAudioPlayback: (...args: unknown[]): Promise<void> =>
    mockSetupAudioPlayback(...args) as Promise<void>,
  stopAudioPlayback: (...args: unknown[]): Promise<void> =>
    mockStopAudioPlayback(...args) as Promise<void>,
  streamMicrophoneToServer: (...args: unknown[]): Promise<void> =>
    mockStreamMicrophoneToServer(...args) as Promise<void>,
  stopMicrophoneStream: (...args: unknown[]): Promise<void> =>
    mockStopMicrophoneStream(...args) as Promise<void>,
  addDataToAudioQueue: (): ReturnType<typeof mockAddDataToAudioQueue> =>
    mockAddDataToAudioQueue(),
}));

const mockIo = vi.fn();
vi.mock("socket.io-client", () => ({
  io: (url: string): ReturnType<typeof mockIo> => mockIo(url),
}));

const mockGetConversationHistory = vi.fn();
vi.mock("../api", () => ({
  getConversationHistory: (...args: unknown[]) =>
    mockGetConversationHistory(...args) as unknown,
}));

vi.mock("../stores/copilot/actions", () => ({
  setSessionId: vi.fn(),
}));

vi.mock("../stores/project/actions", () => ({
  setProjectInactive: vi.fn(),
}));

const mockLogError = vi.fn();
const mockTrack = vi.fn();
const mockShowToast = vi.fn();

vi.mock("../InspectorContext", () => ({
  useInspectorContext: () => ({
    logError: mockLogError,
    track: mockTrack,
    showToast: mockShowToast,
  }),
}));

type MockedUseParams = MockedFunction<typeof useParams>;
type MockedUseLocalStorage = MockedFunction<typeof useLocalStorage>;

type SocketHandler = (...args: unknown[]) => void;
type SocketHandlers = Record<string, SocketHandler>;
type MockSocket = {
  emit: ReturnType<typeof vi.fn>;
  handlers: SocketHandlers;
  on: (event: string, cb: SocketHandler) => void;
  disconnect: ReturnType<typeof vi.fn>;
  removeAllListeners: ReturnType<typeof vi.fn>;
  io: { on: ReturnType<typeof vi.fn>; removeAllListeners: ReturnType<typeof vi.fn> };
};

describe("useBotConnection", () => {
  let lastSocket: MockSocket;

  beforeEach(() => {
    vi.clearAllMocks();
    initInspectorStore();
    mockGetConversationHistory.mockResolvedValue(null);
    (useParams as MockedUseParams).mockReturnValue({
      projectId: "test-project",
    });
    (useLocalStorage as MockedUseLocalStorage).mockReturnValue([
      {},
      vi.fn(),
      vi.fn(),
    ]);
    mockIo.mockImplementation(() => {
      const socket: MockSocket = {
        emit: vi.fn(),
        handlers: {},
        on: (event: string, cb: SocketHandler) => {
          socket.handlers[event] = cb;
        },
        disconnect: vi.fn(),
        removeAllListeners: vi.fn(),
        io: { on: vi.fn(), removeAllListeners: vi.fn() },
      };
      lastSocket = socket;
      return socket;
    });
  });

  it("uses correct localStorage key based on projectId from URL", () => {
    (useLocalStorage as MockedUseLocalStorage).mockReturnValue([
      {},
      vi.fn(),
      vi.fn(),
    ]);

    renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    expect(useLocalStorage).toHaveBeenCalledWith(
      `${REACT_APP_SESSION_HISTORY_KEY}_test-project`,
      {},
    );
  });

  it("syncs initial state to the store with input disabled", () => {
    renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    const state = inspectorStore.state;
    expect(state.inputDisabled).toBe(true);
    expect(state.sessionId).toBeDefined();
    expect(typeof state.sessionId).toBe("string");
    expect(state.sessionId.length).toBeGreaterThan(0);
    expect(state.conversationList).toEqual([]);
  });

  it("startNewConversation updates sessionId in store", () => {
    renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    const initialSessionId = inspectorStore.state.sessionId;

    act(() => {
      inspectorStore.state.startNewConversation();
    });

    expect(inspectorStore.state.sessionId).toBeDefined();
    expect(inspectorStore.state.sessionId).not.toBe(initialSessionId);
  });

  it("startNewConversation clears stack, slots, and slotRelatedEvents", () => {
    renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    act(() => {
      inspectorStore.state.setUrl("https://test.example.com");
    });

    act(() => {
      lastSocket.handlers["connect"]?.();
    });

    act(() => {
      lastSocket.handlers["tracker"]?.({
        sender_id: inspectorStore.state.sessionId,
        events: [],
        slots: [{ name: "some_slot", value: "some_value" }],
        stack: [{ frame_id: "f1", flow_id: "my_flow", step_id: "s1", collect: undefined, utter: undefined }],
      });
    });

    expect(inspectorStore.state.stack).toHaveLength(1);
    expect(inspectorStore.state.slots).toHaveLength(1);

    act(() => {
      inspectorStore.state.startNewConversation();
    });

    expect(inspectorStore.state.stack).toEqual([]);
    expect(inspectorStore.state.slots).toEqual([]);
    expect(inspectorStore.state.slotRelatedEvents).toEqual([]);
  });

  describe("session_start message behavior", () => {
    it("sends /session_start on session_confirm when in text modality", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      act(() => {
        lastSocket.handlers["connect"]?.();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });

      expect(lastSocket.emit).toHaveBeenCalledWith("user_message", {
        message: "/session_start",
        session_id: inspectorStore.state.sessionId,
      });
    });

    it("does not send /session_start on session_confirm when in voice modality", async () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = inspectorStore.state.startVoiceStreaming();
      });

      act(() => {
        lastSocket.handlers["connect"]?.();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.({ session_id: "test-session", sample_rate: 48000 });
      });

      await act(async () => {
        await voicePromise;
      });

      const userMessageCalls = lastSocket.emit.mock.calls.filter(
        (call) => call[0] === "user_message",
      );
      expect(userMessageCalls).toHaveLength(0);
    });
  });

  describe("voice_error event", () => {
    it("calls onVoiceErrorRef callback when voice_error event is received", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      const handler = vi.fn();
      inspectorStore.state.onVoiceErrorRef.current = handler;

      const voiceErrorPayload = {
        message: "Voice streaming failed",
        error: "Missing environment variable for ASR Engine DeepgramASR: DEEPGRAM_API_KEY",
        exception: "ProviderClientValidationError",
      };

      act(() => {
        lastSocket.handlers["voice_error"]?.(voiceErrorPayload);
      });

      expect(handler).toHaveBeenCalledWith(voiceErrorPayload);
    });

    it("does not throw when onVoiceErrorRef.current is null", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      expect(inspectorStore.state.onVoiceErrorRef.current).toBeNull();

      expect(() => {
        act(() => {
          lastSocket.handlers["voice_error"]?.({
            message: "Voice streaming failed",
            error: "some error",
            exception: "SomeException",
          });
        });
      }).not.toThrow();
    });
  });

  describe("waitingForUserInput with sub-agent events", () => {
    function sendTracker(sessionId: string, rawEvents: Record<string, unknown>[]) {
      lastSocket.handlers["tracker"]?.({
        sender_id: sessionId,
        events: rawEvents,
        slots: [],
        stack: [],
      });
    }

    it("returns true when last action is action_agent_request_user_input", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "action", name: "action_agent_request_user_input", timestamp: 1 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(true);
    });

    it("returns true when action_agent_request_user_input is followed by an empty bot utterance", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "action", name: "action_agent_request_user_input", timestamp: 1 },
          { event: "bot", timestamp: 2 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(true);
    });

    it("returns true for action_listen (standard flow)", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "action", name: "action_listen", timestamp: 1 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(true);
    });

    it("returns false when action_listen follows agent_started (sub-agent still working)", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "agent_started", timestamp: 1 },
          { event: "action", name: "action_listen", timestamp: 2 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(false);
    });

    it("returns true when action_listen follows agent_completed (sub-agent done)", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "agent_started", timestamp: 1 },
          { event: "agent_completed", timestamp: 2 },
          { event: "action", name: "action_listen", timestamp: 3 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(true);
    });

    it("returns false when user message follows action_listen (bot is processing)", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "action", name: "action_listen", timestamp: 1 },
          { event: "user", text: "find flights", timestamp: 2 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(false);
    });

    it("returns false when last action is not a listen/request action", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => inspectorStore.state.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(inspectorStore.state.sessionId, [
          { event: "action", name: "action_some_custom", timestamp: 1 },
        ]),
      );

      expect(inspectorStore.state.waitingForUserInput).toBe(false);
    });
  });

  describe("disconnect during voice call", () => {
    it("calls onVoiceErrorRef with connection_lost when disconnected during voice", async () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = inspectorStore.state.startVoiceStreaming();
      });

      act(() => {
        lastSocket.handlers["connect"]?.();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });

      await act(async () => {
        await voicePromise;
      });

      const handler = vi.fn();
      inspectorStore.state.onVoiceErrorRef.current = handler;

      act(() => {
        lastSocket.handlers["disconnect"]?.("transport close", {});
      });

      expect(handler).toHaveBeenCalledWith({
        error: "connection_lost",
        message: "Server connection lost during voice call",
      });
    });

    it("does not call onVoiceErrorRef on disconnect when in text mode, shows toast instead", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      act(() => {
        lastSocket.handlers["connect"]?.();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });

      const handler = vi.fn();
      inspectorStore.state.onVoiceErrorRef.current = handler;

      act(() => {
        lastSocket.handlers["disconnect"]?.("transport close", {});
      });

      expect(handler).not.toHaveBeenCalled();
      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Server disconnected",
        description: "Trying to reconnect...",
        type: "error",
      });
    });
  });

  describe("voice streaming", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it("startVoiceStreaming throws SocketTimeoutError when session_confirm does not arrive within 10s", async () => {
      vi.useFakeTimers();
      renderHook(() => useBotConnection({
        projectId: "test-project",
        onSessionStart: vi.fn(),
        onReconnectError: vi.fn(),
        useMemoryOnly: false,
      }));

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = inspectorStore.state.startVoiceStreaming();
      });

      await act(async () => {
        vi.advanceTimersByTime(10000);
        await expect(voicePromise).rejects.toThrow(SocketTimeoutError);
      });
    });

    it("startVoiceStreaming after session_confirm calls setupAudioPlayback and streamMicrophoneToServer with sample_rate", async () => {
      renderHook(() => useBotConnection({
        projectId: "test-project",
        onSessionStart: vi.fn(),
        onReconnectError: vi.fn(),
        useMemoryOnly: false,
      }));

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = inspectorStore.state.startVoiceStreaming();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.({ session_id: "test-session", sample_rate: 48000 });
      });

      await act(async () => {
        await voicePromise;
      });

      expect(mockCreateAudioQueue).toHaveBeenCalled();
      expect(mockSetupAudioPlayback).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        48000,
        expect.anything(),
      );
      expect(mockStreamMicrophoneToServer).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        48000,
      );
    });

    it("stopVoiceStreaming calls stopMicrophoneStream, stopAudioPlayback and disables chat without starting new conversation", async () => {
      renderHook(() => useBotConnection({
        projectId: "test-project",
        onSessionStart: vi.fn(),
        onReconnectError: vi.fn(),
        useMemoryOnly: false,
      }));

      act(() => {
        inspectorStore.state.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = inspectorStore.state.startVoiceStreaming();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.({ session_id: "test-session", sample_rate: 48000 });
      });
      await act(async () => {
        await voicePromise;
      });

      const sessionIdAfterStart = inspectorStore.state.sessionId;

      await act(async () => {
        await inspectorStore.state.stopVoiceStreaming();
      });

      expect(mockStopMicrophoneStream).toHaveBeenCalled();
      expect(mockStopAudioPlayback).toHaveBeenCalled();
      expect(inspectorStore.state.sessionId).toBe(sessionIdAfterStart);
      expect(inspectorStore.state.inputDisabled).toBe(true);
    });
  });

  describe("when sessionId prop is provided", () => {
    const EXTERNAL_SESSION_ID = "external-session-abc";

    it("uses the provided sessionId as the initial session ID in the store", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          sessionId: EXTERNAL_SESSION_ID,
          useMemoryOnly: true,
        }),
      );

      expect(inspectorStore.state.sessionId).toBe(EXTERNAL_SESSION_ID);
    });

    it("does not send /session_start on session_confirm", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          sessionId: EXTERNAL_SESSION_ID,
          useMemoryOnly: true,
        }),
      );

      act(() => { inspectorStore.state.setUrl("https://test.example.com"); });
      act(() => { lastSocket.handlers["connect"]?.(); });
      act(() => { lastSocket.handlers["session_confirm"]?.(); });

      const sessionStartCalls = lastSocket.emit.mock.calls.filter(
        (call) =>
          call[0] === "user_message" &&
          (call[1] as { message: string }).message === "/session_start",
      );
      expect(sessionStartCalls).toHaveLength(0);
    });

    it("does not call getConversationHistory when trackerEndpoint is not set in the store", () => {
      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          sessionId: EXTERNAL_SESSION_ID,
          useMemoryOnly: true,
        }),
      );

      act(() => { inspectorStore.state.setUrl("https://test.example.com"); });
      act(() => { lastSocket.handlers["connect"]?.(); });
      act(() => { lastSocket.handlers["session_confirm"]?.(); });

      expect(mockGetConversationHistory).not.toHaveBeenCalled();
    });

    it("calls getConversationHistory with projectUrl and trackerEndpoint on session_confirm", () => {
      initInspectorStore({
        trackerEndpoint: "/tracker",
        projectUrl: "https://test.example.com",
      });

      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          sessionId: EXTERNAL_SESSION_ID,
          useMemoryOnly: true,
        }),
      );

      act(() => { inspectorStore.state.setUrl("https://test.example.com"); });
      act(() => { lastSocket.handlers["connect"]?.(); });
      act(() => { lastSocket.handlers["session_confirm"]?.(); });

      expect(mockGetConversationHistory).toHaveBeenCalledWith({
        projectUrl: "https://test.example.com",
        trackerEndpoint: "/tracker",
      });
    });

    it("calls logError when getConversationHistory rejects", async () => {
      const fetchError = new Error("network error");
      initInspectorStore({
        trackerEndpoint: "/tracker",
        projectUrl: "https://test.example.com",
      });
      mockGetConversationHistory.mockRejectedValue(fetchError);

      renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          sessionId: EXTERNAL_SESSION_ID,
          useMemoryOnly: true,
        }),
      );

      act(() => { inspectorStore.state.setUrl("https://test.example.com"); });
      act(() => { lastSocket.handlers["connect"]?.(); });

      await act(async () => {
        lastSocket.handlers["session_confirm"]?.();
        await Promise.resolve();
      });

      expect(mockLogError).toHaveBeenCalledWith(fetchError);
    });
  });
});
