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

vi.mock("react-router-dom", () => ({
  useParams: vi.fn(),
}));

vi.mock("../hooks/useLocalStorage", () => ({
  useLocalStorage: vi.fn(),
}));

const mockCreateAudioQueue = vi.fn(() => ({ buffer: [], socket: {} }));
const mockSetupAudioPlayback = vi.fn().mockResolvedValue(undefined);
const mockStopAudioPlayback = vi.fn().mockResolvedValue(undefined);
const mockStreamMicrophoneToServer = vi.fn().mockResolvedValue(undefined);
const mockStopMicrophoneStream = vi.fn().mockResolvedValue(undefined);
const mockAddDataToAudioQueue = vi.fn(() => vi.fn());

vi.mock("../utils/voice/audiostream", () => ({
  createAudioQueue: () => mockCreateAudioQueue(),
  setupAudioPlayback: (): Promise<void> =>
    mockSetupAudioPlayback() as Promise<void>,
  stopAudioPlayback: (): Promise<void> =>
    mockStopAudioPlayback() as Promise<void>,
  streamMicrophoneToServer: (): Promise<void> =>
    mockStreamMicrophoneToServer() as Promise<void>,
  stopMicrophoneStream: (): Promise<void> =>
    mockStopMicrophoneStream() as Promise<void>,
  addDataToAudioQueue: (): ReturnType<typeof mockAddDataToAudioQueue> =>
    mockAddDataToAudioQueue(),
}));

const mockIo = vi.fn();
vi.mock("socket.io-client", () => ({
  io: (url: string): ReturnType<typeof mockIo> => mockIo(url),
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
        handlers: {} as SocketHandlers,
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

  it("returns initial state with input disabled and valid sessionId", () => {
    const { result } = renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    expect(result.current.inputDisabled).toBe(true);
    expect(result.current.sessionId).toBeDefined();
    expect(typeof result.current.sessionId).toBe("string");
    expect(result.current.sessionId.length).toBeGreaterThan(0);
    expect(result.current.conversationList).toEqual([]);
    expect(result.current.error).toBeUndefined();
  });

  it("startNewConversation returns new sessionId and updates sessionId", () => {
    const { result } = renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    const initialSessionId = result.current.sessionId;
    let newSessionId: string | undefined;

    act(() => {
      newSessionId = result.current.startNewConversation();
    });

    expect(newSessionId).toBeDefined();
    expect(newSessionId).not.toBe(initialSessionId);
    expect(result.current.sessionId).toBe(newSessionId);
  });

  it("startNewConversation clears stack, slots, and slotRelatedEvents", () => {
    const { result } = renderHook(() => useBotConnection({
      projectId: "test-project",
      onSessionStart: vi.fn(),
      onReconnectError: vi.fn(),
      useMemoryOnly: false,
    }));

    act(() => {
      result.current.setUrl("https://test.example.com");
    });

    act(() => {
      lastSocket.handlers["connect"]?.();
    });

    act(() => {
      lastSocket.handlers["tracker"]?.({
        sender_id: result.current.sessionId,
        events: [],
        slots: [{ name: "some_slot", value: "some_value" }],
        stack: [{ frame_id: "f1", flow_id: "my_flow", step_id: "s1", collect: undefined, utter: undefined }],
      });
    });

    expect(result.current.stack).toHaveLength(1);
    expect(result.current.slots).toHaveLength(1);

    act(() => {
      result.current.startNewConversation();
    });

    expect(result.current.stack).toEqual([]);
    expect(result.current.slots).toEqual([]);
    expect(result.current.slotRelatedEvents).toEqual([]);
  });

  describe("session_start message behavior", () => {
    it("sends /session_start on session_confirm when in text modality", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      act(() => {
        lastSocket.handlers["connect"]?.();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });

      expect(lastSocket.emit).toHaveBeenCalledWith("user_message", {
        message: "/session_start",
        session_id: result.current.sessionId,
      });
    });

    it("does not send /session_start on session_confirm when in voice modality", async () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = result.current.startVoiceStreaming();
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

      const userMessageCalls = lastSocket.emit.mock.calls.filter(
        (call) => call[0] === "user_message",
      );
      expect(userMessageCalls).toHaveLength(0);
    });
  });

  describe("voice_error event", () => {
    it("calls onVoiceErrorRef callback when voice_error event is received", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      const handler = vi.fn();
      result.current.onVoiceErrorRef.current = handler;

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
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      expect(result.current.onVoiceErrorRef.current).toBeNull();

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
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "action", name: "action_agent_request_user_input", timestamp: 1 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(true);
    });

    it("returns true when action_agent_request_user_input is followed by an empty bot utterance", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "action", name: "action_agent_request_user_input", timestamp: 1 },
          { event: "bot", timestamp: 2 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(true);
    });

    it("returns true for action_listen (standard flow)", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "action", name: "action_listen", timestamp: 1 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(true);
    });

    it("returns false when action_listen follows agent_started (sub-agent still working)", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "agent_started", timestamp: 1 },
          { event: "action", name: "action_listen", timestamp: 2 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(false);
    });

    it("returns true when action_listen follows agent_completed (sub-agent done)", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "agent_started", timestamp: 1 },
          { event: "agent_completed", timestamp: 2 },
          { event: "action", name: "action_listen", timestamp: 3 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(true);
    });

    it("returns false when user message follows action_listen (bot is processing)", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "action", name: "action_listen", timestamp: 1 },
          { event: "user", text: "find flights", timestamp: 2 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(false);
    });

    it("returns false when last action is not a listen/request action", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: false,
        }),
      );

      act(() => result.current.setUrl("https://test.example.com"));
      act(() => lastSocket.handlers["connect"]?.());

      act(() =>
        sendTracker(result.current.sessionId, [
          { event: "action", name: "action_some_custom", timestamp: 1 },
        ]),
      );

      expect(result.current.waitingForUserInput).toBe(false);
    });
  });

  describe("disconnect during voice call", () => {
    it("calls onVoiceErrorRef with connection_lost when disconnected during voice", async () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = result.current.startVoiceStreaming();
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
      result.current.onVoiceErrorRef.current = handler;

      act(() => {
        lastSocket.handlers["disconnect"]?.("transport close", {});
      });

      expect(handler).toHaveBeenCalledWith({
        error: "connection_lost",
        message: "Server connection lost during voice call",
      });
    });

    it("does not call onVoiceErrorRef on disconnect when in text mode, shows toast instead", () => {
      const { result } = renderHook(() =>
        useBotConnection({
          projectId: "test-project",
          onSessionStart: vi.fn(),
          onReconnectError: vi.fn(),
          useMemoryOnly: true,
        }),
      );

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      act(() => {
        lastSocket.handlers["connect"]?.();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });

      const handler = vi.fn();
      result.current.onVoiceErrorRef.current = handler;

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
      const { result } = renderHook(() => useBotConnection({
        projectId: "test-project",
        onSessionStart: vi.fn(),
        onReconnectError: vi.fn(),
        useMemoryOnly: false,
      }));

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = result.current.startVoiceStreaming();
      });

      await act(async () => {
        vi.advanceTimersByTime(10000);
        await expect(voicePromise).rejects.toThrow(SocketTimeoutError);
      });
    });

    it("startVoiceStreaming after session_confirm calls setupAudioPlayback and streamMicrophoneToServer", async () => {
      const { result } = renderHook(() => useBotConnection({
        projectId: "test-project",
        onSessionStart: vi.fn(),
        onReconnectError: vi.fn(),
        useMemoryOnly: false,
      }));

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = result.current.startVoiceStreaming();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });

      await act(async () => {
        await voicePromise;
      });

      expect(mockCreateAudioQueue).toHaveBeenCalled();
      expect(mockSetupAudioPlayback).toHaveBeenCalled();
      expect(mockStreamMicrophoneToServer).toHaveBeenCalled();
    });

    it("stopVoiceStreaming calls stopMicrophoneStream, stopAudioPlayback and startNewConversation", async () => {
      const { result } = renderHook(() => useBotConnection({
        projectId: "test-project",
        onSessionStart: vi.fn(),
        onReconnectError: vi.fn(),
        useMemoryOnly: false,
      }));

      act(() => {
        result.current.setUrl("https://test.example.com");
      });

      let voicePromise: Promise<void>;
      act(() => {
        voicePromise = result.current.startVoiceStreaming();
      });

      act(() => {
        lastSocket.handlers["session_confirm"]?.();
      });
      await act(async () => {
        await voicePromise;
      });

      const sessionIdAfterStart = result.current.sessionId;

      await act(async () => {
        await result.current.stopVoiceStreaming();
      });

      expect(mockStopMicrophoneStream).toHaveBeenCalled();
      expect(mockStopAudioPlayback).toHaveBeenCalled();
      expect(result.current.sessionId).not.toBe(sessionIdAfterStart);
    });
  });
});
