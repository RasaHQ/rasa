import { screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../tests/utils";
import type { VoiceErrorHandler } from "../types";
import { MessageInput } from "./MessageInput";

const mockStartVoiceStreaming = vi.fn().mockResolvedValue(undefined);
const mockStopVoiceStreaming = vi.fn().mockResolvedValue(undefined);
const mockOnVoiceErrorRef: { current: VoiceErrorHandler } = { current: null };

vi.mock("../../hooks/useVoiceCall", () => ({
  useVoiceCall: vi.fn(() => ({
    callDuration: "00:00",
    startVoiceCall: mockStartVoiceStreaming,
    stopVoiceCall: mockStopVoiceStreaming,
    voiceCallState: "inactive",
  })),
}));

describe("MessageInput", () => {
  const onSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders input with placeholder and correct action button for empty vs typed text", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MessageInput
        onSubmit={onSubmit}
        startVoiceStreaming={mockStartVoiceStreaming}
        stopVoiceStreaming={mockStopVoiceStreaming}
        onVoiceErrorRef={mockOnVoiceErrorRef}
        voiceFeaturesEnabled={true}
      />,
    );

    // Input is empty and voice button is shown
    expect(screen.getByTestId("assistant-input")).toBeInTheDocument();
    const input = screen.getByPlaceholderText("Type your message");
    expect(input).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /start voice conversation/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /send message/i }),
    ).not.toBeInTheDocument();

    // Input is not empty and send button is shown
    await user.type(input, "Hello");
    expect(
      screen.getByRole("button", { name: /send message/i }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /start voice conversation/i }),
    ).not.toBeInTheDocument();

    // Clicking send button submits the message
    await user.click(screen.getByRole("button", { name: /send message/i }));
    expect(onSubmit).toHaveBeenCalledWith("Hello");
    expect(input).toHaveValue("");
  });

  it("submits on Enter key", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MessageInput
        onSubmit={onSubmit}
        startVoiceStreaming={mockStartVoiceStreaming}
        stopVoiceStreaming={mockStopVoiceStreaming}
        onVoiceErrorRef={mockOnVoiceErrorRef}
        voiceFeaturesEnabled={true}
      />,
    );

    const input = screen.getByPlaceholderText("Type your message");
    await user.type(input, "Hi{Enter}");

    expect(onSubmit).toHaveBeenCalledWith("Hi");
  });

  it("input and voice button are disabled when isDisabled is true", () => {
    renderWithProviders(
      <MessageInput
        onSubmit={onSubmit}
        isDisabled={true}
        startVoiceStreaming={mockStartVoiceStreaming}
        stopVoiceStreaming={mockStopVoiceStreaming}
        onVoiceErrorRef={mockOnVoiceErrorRef}
        voiceFeaturesEnabled={true}
      />,
    );

    const input = screen.getByPlaceholderText("Type your message");
    expect(input).toBeDisabled();
    expect(
      screen.getByRole("button", { name: /start voice conversation/i }),
    ).toBeDisabled();
  });

  describe("when voiceFeaturesEnabled is false", () => {
    it("shows only send button, no voice UI, and submit works", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <MessageInput
          onSubmit={onSubmit}
          startVoiceStreaming={mockStartVoiceStreaming}
          stopVoiceStreaming={mockStopVoiceStreaming}
          onVoiceErrorRef={mockOnVoiceErrorRef}
          voiceFeaturesEnabled={false}
        />,
      );

      expect(screen.getByTestId("assistant-input")).toBeInTheDocument();
      expect(
        screen.getByPlaceholderText("Type your message"),
      ).toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: /start voice conversation/i }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: /stop voice conversation/i }),
      ).not.toBeInTheDocument();

      const sendButton = screen.getByRole("button", { name: /send message/i });
      expect(sendButton).toBeDisabled();

      const input = screen.getByPlaceholderText("Type your message");
      await user.type(input, "Hello");
      expect(sendButton).not.toBeDisabled();
      await user.click(sendButton);

      expect(onSubmit).toHaveBeenCalledWith("Hello");
      expect(input).toHaveValue("");
    });
  });
});
