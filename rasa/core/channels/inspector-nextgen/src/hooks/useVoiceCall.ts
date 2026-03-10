import { useCallback, useEffect, useRef, useState } from "react";
import { toaster } from "../Toaster";
import { useInspectorContext } from "../InspectorContext";
import { SocketTimeoutError, SocketUnavailableError } from "../errors";

export type VoiceCallState = "active" | "inactive" | "connecting";

interface Props {
  startVoiceStreaming: () => Promise<void>;
  stopVoiceStreaming: () => Promise<void>;
  voiceFeaturesEnabled: boolean;
}

export const useVoiceCall = ({
  startVoiceStreaming,
  stopVoiceStreaming,
  voiceFeaturesEnabled,
}: Props): {
  callDuration: string;
  startVoiceCall: () => Promise<void>;
  stopVoiceCall: () => Promise<void>;
  voiceCallState: VoiceCallState;
} => {
  const { logError, track } = useInspectorContext();
  const [voiceCallState, setVoiceCallState] =
    useState<VoiceCallState>("inactive");
  const voiceCallTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const operationRef = useRef<"start" | "stop" | null>(null);
  const [callDuration, setCallDuration] = useState<string>("00:00");
  const callDurationRef = useRef<number>(0);

  const stopVoiceCall = useCallback(async () => {
    if (operationRef.current === "stop") return;
    operationRef.current = "stop";

    try {
      if (voiceCallTimerRef.current) {
        clearInterval(voiceCallTimerRef.current);
        voiceCallTimerRef.current = null;
      }
      await stopVoiceStreaming();
    } finally {
      operationRef.current = null;
      setVoiceCallState("inactive");
      callDurationRef.current = 0;
      setCallDuration("00:00");
    }
  }, [stopVoiceStreaming]);

  const startVoiceCall = useCallback(async () => {
    void track("Voice Call Initiated", { target: "rasa_agent" });
    if (operationRef.current !== null || voiceCallTimerRef.current) return;
    operationRef.current = "start";

    setCallDuration("00:00");
    setVoiceCallState("connecting");
    callDurationRef.current = 0;
    try {
      await startVoiceStreaming();
      void track("Voice Call Started", { target: "rasa_agent" });
    } catch (error) {
      const userErrorMessage = getUserErrorMessage(error);
      if (!isRecognizedError(error)) {
        logError(error, {
          tags: { component: "useVoiceCall", action: "startVoiceCall" },
        });
      }
      toaster.create({
        title: "Failed to start voice call",
        description: userErrorMessage,
        type: "error",
        duration: 5000,
      });
      setVoiceCallState("inactive");
      await stopVoiceCall();
      return;
    }

    if (voiceCallTimerRef.current) {
      clearInterval(voiceCallTimerRef.current);
      voiceCallTimerRef.current = null;
    }
    voiceCallTimerRef.current = setInterval(() => {
      callDurationRef.current += 1;
      setCallDuration(formatDuration(callDurationRef.current));
    }, 1000);
    setVoiceCallState("active");
  }, [startVoiceStreaming, stopVoiceCall, logError, track]);

  useEffect(() => {
    return () => {
      void stopVoiceCall();
    };
  }, [stopVoiceCall]);

  return voiceFeaturesEnabled
    ? {
      callDuration,
      startVoiceCall,
      stopVoiceCall,
      voiceCallState,
    }
    : {
      callDuration: "00:00",
      startVoiceCall: () => Promise.resolve(),
      stopVoiceCall: () => Promise.resolve(),
      voiceCallState: "inactive",
    };
};

// Format duration in the format of "00:00"
// If call duration is more than 1 hour, show it in the format of "01:00:00"
const formatDuration = (duration: number) => {
  const hours = Math.floor(duration / 3600);
  const minutes = Math.floor((duration % 3600) / 60);
  const seconds = duration % 60;
  if (hours === 0) {
    return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  }
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
};

const isRecognizedError = (error: unknown): boolean =>
  error instanceof SocketTimeoutError ||
  error instanceof SocketUnavailableError ||
  error instanceof Error;

const getUserErrorMessage = (error: unknown): string => {
  if (error instanceof SocketTimeoutError) {
    return "Connection timeout - please try again";
  } else if (error instanceof SocketUnavailableError) {
    return "Connection unavailable - please try again";
  } else if (error instanceof Error) {
    switch (error.name) {
      case "NotAllowedError":
        return "Microphone permission denied";
      case "NotReadableError":
        return "Microphone is not available";
      case "NotFoundError":
        return "Microphone is not found";
      default:
        return "Failed to start voice call";
    }
  }
  return "An unknown error occurred";
};
