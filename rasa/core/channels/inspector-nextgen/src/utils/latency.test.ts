import { describe, expect, it } from "vitest";
import { isRasaExecutionTime, isVoiceLatency } from "./latency";

describe("isVoiceLatency", () => {
  it("returns true for a complete voice latency object", () => {
    expect(
      isVoiceLatency({
        asr_latency_ms: 1,
        rasa_processing_latency_ms: 2,
        tts_complete_latency_ms: 3,
        tts_first_byte_latency_ms: 4,
      }),
    ).toBe(true);
  });

  it("returns false when a field is missing or not a number", () => {
    expect(isVoiceLatency(null)).toBe(false);
    expect(isVoiceLatency({})).toBe(false);
    expect(
      isVoiceLatency({
        asr_latency_ms: 1,
        rasa_processing_latency_ms: 2,
        tts_complete_latency_ms: 3,
      }),
    ).toBe(false);
    expect(
      isVoiceLatency({
        asr_latency_ms: "1",
        rasa_processing_latency_ms: 2,
        tts_complete_latency_ms: 3,
        tts_first_byte_latency_ms: 4,
      }),
    ).toBe(false);
  });
});

describe("isRasaExecutionTime", () => {
  it("returns true for valid execution_times shape", () => {
    expect(
      isRasaExecutionTime({
        command_processor: 100,
        prediction_loop: 200,
      }),
    ).toBe(true);
  });

  it("returns false for invalid values", () => {
    expect(isRasaExecutionTime(null)).toBe(false);
    expect(isRasaExecutionTime({ command_processor: 1 })).toBe(false);
    expect(
      isRasaExecutionTime({
        command_processor: "1",
        prediction_loop: 2,
      }),
    ).toBe(false);
  });
});
