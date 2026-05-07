import type { RasaExecutionTime, VoiceLatency } from "../types/conversation";

export function isVoiceLatency(latency: unknown): latency is VoiceLatency {
  return (
    typeof latency === "object" &&
    latency !== null &&
    "rasa_processing_latency_ms" in latency &&
    "asr_latency_ms" in latency &&
    "tts_complete_latency_ms" in latency &&
    "tts_first_byte_latency_ms" in latency &&
    typeof (latency as VoiceLatency).asr_latency_ms === "number" &&
    typeof (latency as VoiceLatency).rasa_processing_latency_ms === "number" &&
    typeof (latency as VoiceLatency).tts_complete_latency_ms === "number" &&
    typeof (latency as VoiceLatency).tts_first_byte_latency_ms === "number"
  );
}

export function isRasaExecutionTime(
  executionTime: unknown,
): executionTime is RasaExecutionTime {
  return (
    typeof executionTime === "object" &&
    executionTime !== null &&
    "command_processor" in executionTime &&
    "prediction_loop" in executionTime &&
    typeof (executionTime as RasaExecutionTime).command_processor === "number" &&
    typeof (executionTime as RasaExecutionTime).prediction_loop === "number"
  );
}

export function botUtteranceHasLatencyMetadata(utterance: {
  metadata?: { execution_times?: unknown; voiceLatency?: unknown };
}): boolean {
  return (
    isVoiceLatency(utterance.metadata?.voiceLatency) ||
    isRasaExecutionTime(utterance.metadata?.execution_times)
  );
}
