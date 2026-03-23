import { renderHook, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi, afterEach } from "vitest";
import { useVoiceCall } from "./useVoiceCall";
import { SocketTimeoutError, SocketUnavailableError } from "../errors";
import type { VoiceErrorHandler } from "../types";

const mockLogError = vi.fn();
const mockShowToast = vi.fn();

vi.mock("../InspectorContext", () => ({
  useInspectorContext: () => ({
    logError: mockLogError,
    track: vi.fn(),
    showToast: mockShowToast,
  }),
}));

describe("useVoiceCall", () => {
  let startVoiceStreaming: ReturnType<typeof vi.fn>;
  let stopVoiceStreaming: ReturnType<typeof vi.fn>;
  let onVoiceErrorRef: { current: VoiceErrorHandler };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
    startVoiceStreaming = vi.fn().mockResolvedValue(undefined);
    stopVoiceStreaming = vi.fn().mockResolvedValue(undefined);
    onVoiceErrorRef = { current: null };
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should initialize with inactive state, 00:00 duration, and function references", () => {
    const { result } = renderHook(() =>
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
    );

    expect(result.current.voiceCallState).toBe("inactive");
    expect(result.current.callDuration).toBe("00:00");
    expect(typeof result.current.startVoiceCall).toBe("function");
    expect(typeof result.current.stopVoiceCall).toBe("function");
  });

  it("should transition from inactive to active and back to inactive when stopped", async () => {
    const { result } = renderHook(() =>
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
    );

    expect(result.current.voiceCallState).toBe("inactive");

    await act(async () => {
      await result.current.startVoiceCall();
    });

    expect(result.current.voiceCallState).toBe("active");
    expect(startVoiceStreaming).toHaveBeenCalledTimes(1);

    await act(async () => {
      await result.current.stopVoiceCall();
    });

    expect(result.current.voiceCallState).toBe("inactive");
    expect(result.current.callDuration).toBe("00:00");
    expect(stopVoiceStreaming).toHaveBeenCalledTimes(1);
  });

  describe("call duration timer", () => {
    it("should increment every second and format correctly (minutes, hours)", async () => {
      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await act(async () => {
        await result.current.startVoiceCall();
      });

      expect(result.current.callDuration).toBe("00:00");

      // Test seconds increment with leading zeros
      act(() => {
        vi.advanceTimersByTime(1000);
      });
      expect(result.current.callDuration).toBe("00:01");

      act(() => {
        vi.advanceTimersByTime(8000);
      });
      expect(result.current.callDuration).toBe("00:09");

      // Test minutes
      act(() => {
        vi.advanceTimersByTime(51000); // 60 total
      });
      expect(result.current.callDuration).toBe("01:00");

      // Test hours format
      act(() => {
        vi.advanceTimersByTime(3540000); // 1 hour total
      });
      expect(result.current.callDuration).toBe("01:00:00");

      // Test multi-digit hours (from 1:00:00 to 10:15:30 = 9h 15m 30s = 33330s)
      act(() => {
        vi.advanceTimersByTime(33330000);
      });
      expect(result.current.callDuration).toBe("10:15:30");
    });

    it("should stop timer when call is stopped and reset on new call", async () => {
      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await act(async () => {
        await result.current.startVoiceCall();
      });

      act(() => {
        vi.advanceTimersByTime(5000);
      });
      expect(result.current.callDuration).toBe("00:05");

      await act(async () => {
        await result.current.stopVoiceCall();
      });
      expect(result.current.callDuration).toBe("00:00");

      // Timer should not advance anymore
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      expect(result.current.callDuration).toBe("00:00");

      // Reset on new call
      await act(async () => {
        await result.current.startVoiceCall();
      });
      expect(result.current.callDuration).toBe("00:00");

      // Timer should advance again
      act(() => {
        vi.advanceTimersByTime(5000);
      });
      expect(result.current.callDuration).toBe("00:05");
    });
  });

  describe("error handling", () => {
    it("should handle SocketTimeoutError and call stopVoiceStreaming", async () => {
      startVoiceStreaming.mockRejectedValueOnce(
        new SocketTimeoutError("Timeout"),
      );

      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await result.current.startVoiceCall();

      expect(result.current.voiceCallState).toBe("inactive");
      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Failed to start voice call",
        description: "Connection timeout - please try again",
        type: "error",
        duration: 5000,
      });
      expect(stopVoiceStreaming).toHaveBeenCalledTimes(1);
    });

    it("should handle SocketUnavailableError", async () => {
      startVoiceStreaming.mockRejectedValueOnce(
        new SocketUnavailableError("Unavailable"),
      );

      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await result.current.startVoiceCall();

      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Failed to start voice call",
        description: "Connection unavailable - please try again",
        type: "error",
        duration: 5000,
      });
    });

    it.each([
      { name: "NotAllowedError", description: "Microphone permission denied" },
      { name: "NotReadableError", description: "Microphone is not available" },
      { name: "NotFoundError", description: "Microphone is not found" },
    ])(
      "should handle $name with message: $description",
      async ({ name, description }) => {
        const error = new Error("Media error");
        error.name = name;
        startVoiceStreaming.mockRejectedValueOnce(error);

        const { result } = renderHook(() =>
          useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
        );

        await result.current.startVoiceCall();

        expect(mockShowToast).toHaveBeenCalledWith({
          title: "Failed to start voice call",
          description,
          type: "error",
          duration: 5000,
        });
      },
    );

    it("should handle generic errors and non-Error objects", async () => {
      // Generic Error
      startVoiceStreaming.mockRejectedValueOnce(new Error("Some error"));

      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await result.current.startVoiceCall();

      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Failed to start voice call",
        description: "Failed to start voice call",
        type: "error",
        duration: 5000,
      });

      // Non-Error object
      const nonError = { message: "Not an Error object" };
      startVoiceStreaming.mockRejectedValueOnce(nonError);

      await result.current.startVoiceCall();

      expect(mockLogError).toHaveBeenCalledWith(nonError, {
        tags: {
          component: "useVoiceCall",
          action: "startVoiceCall",
        },
      });

      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Failed to start voice call",
        description: "An unknown error occurred",
        type: "error",
        duration: 5000,
      });
    });

    it("should stop voice call and show toast when voiceError callback is invoked", async () => {
      const voiceError = {
        message: "Voice streaming failed",
        error: "Missing environment variable for ASR Engine DeepgramASR: DEEPGRAM_API_KEY",
        exception: "ProviderClientValidationError",
      };

      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await act(async () => {
        await result.current.startVoiceCall();
      });
      expect(result.current.voiceCallState).toBe("active");
      expect(onVoiceErrorRef.current).toBeTypeOf("function");

      act(() => {
        onVoiceErrorRef.current!(voiceError);
      });

      // eslint-disable-next-line @typescript-eslint/no-empty-function
      await act(async () => {});

      expect(result.current.voiceCallState).toBe("inactive");
      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Voice isn't set up yet",
        description: "To test in voice, add your Voice API keys and complete the voice configuration.",
        type: "warning",
        closable: true,
      });
      expect(stopVoiceStreaming).toHaveBeenCalled();
    });

    it("should stop voice call and show connection lost toast on server disconnect", async () => {
      const connectionLostError = {
        message: "Server connection lost during voice call",
        error: "connection_lost",
      };

      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await act(async () => {
        await result.current.startVoiceCall();
      });
      expect(result.current.voiceCallState).toBe("active");

      act(() => {
        onVoiceErrorRef.current!(connectionLostError);
      });

      // eslint-disable-next-line @typescript-eslint/no-empty-function
      await act(async () => {});

      expect(result.current.voiceCallState).toBe("inactive");
      expect(result.current.callDuration).toBe("00:00");
      expect(mockShowToast).toHaveBeenCalledWith({
        title: "Voice call ended",
        description: "The server connection was lost.",
        type: "error",
        duration: 5000,
      });
      expect(stopVoiceStreaming).toHaveBeenCalled();
    });

    it("should not start timer if error occurs", async () => {
      startVoiceStreaming.mockRejectedValueOnce(new Error("Failed"));

      const { result } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await act(async () => {
        await result.current.startVoiceCall();
      });

      expect(result.current.voiceCallState).toBe("inactive");

      act(() => {
        vi.advanceTimersByTime(5000);
      });
      expect(result.current.callDuration).toBe("00:00");
    });
  });

  it("should guard against concurrent startVoiceCall (no interval leak)", async () => {
    const { result } = renderHook(() =>
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
    );

    await act(async () => {
      await Promise.all([
        result.current.startVoiceCall(),
        result.current.startVoiceCall(),
      ]);
    });

    expect(startVoiceStreaming).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(result.current.callDuration).toBe("00:05");
  });

  it("should guard against concurrent stopVoiceCall", async () => {
    const { result } = renderHook(() =>
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
    );

    await act(async () => {
      await result.current.startVoiceCall();
    });
    expect(result.current.voiceCallState).toBe("active");

    await act(async () => {
      await Promise.all([
        result.current.stopVoiceCall(),
        result.current.stopVoiceCall(),
      ]);
    });

    expect(result.current.voiceCallState).toBe("inactive");
    expect(stopVoiceStreaming).toHaveBeenCalledTimes(1);
  });

  it("should handle rapid start/stop and double calls", async () => {
    const { result } = renderHook(() =>
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
    );

    // Concurrent double start - second call is ignored (no interval leak)
    await act(async () => {
      await Promise.all([
        result.current.startVoiceCall(),
        result.current.startVoiceCall(),
      ]);
    });

    expect(result.current.voiceCallState).toBe("active");
    expect(startVoiceStreaming).toHaveBeenCalledTimes(1);

    // Rapid stop after start
    await act(async () => {
      await result.current.stopVoiceCall();
    });

    expect(stopVoiceStreaming).toHaveBeenCalled();
    expect(result.current.voiceCallState).toBe("inactive");

    // Stop without active call
    await act(async () => {
      await result.current.stopVoiceCall();
    });

    expect(result.current.voiceCallState).toBe("inactive");
  });

  describe("cleanup on unmount", () => {
    it("should call stopVoiceStreaming and clear timer when unmounting with active call", async () => {
      const { result, unmount } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      await act(async () => {
        await result.current.startVoiceCall();
      });

      act(() => {
        vi.advanceTimersByTime(5000);
      });
      expect(result.current.callDuration).toBe("00:05");

      unmount();

      expect(stopVoiceStreaming).toHaveBeenCalledTimes(1);

      // Timer should be cleared - advancing time after unmount should not throw
      expect(() => {
        act(() => {
          vi.advanceTimersByTime(5000);
        });
      }).not.toThrow();
    });

    it("should call stopVoiceStreaming on unmount even when voice was never started", () => {
      const { unmount } = renderHook(() =>
        useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
      );

      unmount();

      expect(stopVoiceStreaming).toHaveBeenCalledTimes(1);
    });
  });

  it("should handle stopVoiceStreaming errors gracefully", async () => {
    stopVoiceStreaming.mockRejectedValueOnce(new Error("Stop failed"));

    const { result } = renderHook(() =>
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled: true }),
    );

    await result.current.startVoiceCall();

    await expect(result.current.stopVoiceCall()).rejects.toThrow("Stop failed");

    expect(result.current.voiceCallState).toBe("inactive");
  });
});
