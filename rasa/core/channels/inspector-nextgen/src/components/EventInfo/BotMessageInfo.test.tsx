import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { Utterance } from "../../types";
import { UtteranceType } from "../../types/conversation";
import { renderWithProviders } from "../../tests/utils";
import { BotMessageInfo } from "./BotMessageInfo";

vi.mock("./DetailView", () => ({
  DetailView: ({
    title,
    onClose,
    children,
  }: {
    title: string;
    onClose: () => void;
    children: React.ReactNode;
  }) => (
    <div>
      <h1>{title}</h1>
      <button data-testid="event-details-close" onClick={onClose}>
        Close
      </button>
      {children}
    </div>
  ),
}));

function baseUtterance(overrides: Partial<Utterance> = {}): Utterance {
  return {
    __typename: "Utterance",
    id: "u1",
    metadata: {
      parseData: {},
    },
    rephrase: false,
    rephrasePrompt: null,
    text: "Hello",
    timestamp: "2020-01-01T00:00:00.000Z",
    tokens: [],
    type: UtteranceType.Bot,
    originalTimestamp: 0,
    ...overrides,
  };
}

function renderBotMessageInfo(utterance: Utterance) {
  return renderWithProviders(
    <BotMessageInfo utterance={utterance} onClose={() => undefined} />,
  );
}

describe("BotMessageInfo", () => {
  it("uses Agent response details as the panel title", () => {
    renderBotMessageInfo(baseUtterance());
    expect(
      screen.getByRole("heading", { name: /agent response details/i }),
    ).toBeInTheDocument();
  });

  it("shows latency accordion with voice headline when voiceLatency is present", () => {
    renderBotMessageInfo(
      baseUtterance({
        metadata: {
          parseData: {},
          execution_times: {
            command_processor: 10,
            prediction_loop: 20,
          },
          voiceLatency: {
            asr_latency_ms: 100,
            rasa_processing_latency_ms: 200,
            tts_complete_latency_ms: 300,
            tts_first_byte_latency_ms: 50,
          },
        },
      }),
    );
    expect(screen.getByText("Latency per turn:")).toBeInTheDocument();
    expect(screen.getByText("~250 ms")).toBeInTheDocument();
    expect(screen.getByText("TTS First Byte:")).toBeInTheDocument();
  });

  it("shows latency from execution_times when voiceLatency is absent", () => {
    renderBotMessageInfo(
      baseUtterance({
        metadata: {
          parseData: {},
          execution_times: {
            command_processor: 40,
            prediction_loop: 60,
          },
        },
      }),
    );
    expect(screen.getByText("Latency per turn:")).toBeInTheDocument();
    expect(screen.getByText("~100 ms")).toBeInTheDocument();
    expect(screen.getByText("Command processor:")).toBeInTheDocument();
  });
});
