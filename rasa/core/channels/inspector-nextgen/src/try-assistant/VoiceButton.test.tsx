import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { renderWithProviders } from "../tests/utils";
import { VoiceButton } from "./VoiceButton";

describe("VoiceButton", () => {
  const startCall = vi.fn().mockResolvedValue(undefined);
  const stopCall = vi.fn().mockResolvedValue(undefined);

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it.each([
    {
      state: "inactive" as const,
      expectedLabel: /start voice conversation/i,
      shouldBeDisabled: false,
    },
    {
      state: "connecting" as const,
      expectedLabel: /start voice conversation/i,
      shouldBeDisabled: true,
    },
    {
      state: "active" as const,
      expectedLabel: /stop voice conversation/i,
      shouldBeDisabled: false,
    },
  ])(
    "should render correct button for $state state",
    ({ state, expectedLabel, shouldBeDisabled }) => {
      renderWithProviders(
        <VoiceButton
          voiceCallState={state}
          startCall={startCall}
          stopCall={stopCall}
        />,
      );

      const button = screen.getByRole("button", { name: expectedLabel });
      expect(button).toBeInTheDocument();

      if (shouldBeDisabled) {
        expect(button).toBeDisabled();
      } else {
        expect(button).not.toBeDisabled();
      }
    },
  );

  it("should call startCall when inactive button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <VoiceButton
        voiceCallState="inactive"
        startCall={startCall}
        stopCall={stopCall}
      />,
    );

    const button = screen.getByRole("button", {
      name: /start voice conversation/i,
    });
    await user.click(button);

    expect(startCall).toHaveBeenCalledOnce();
    expect(stopCall).not.toHaveBeenCalled();
  });

  it("should call stopCall when active button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <VoiceButton
        voiceCallState="active"
        startCall={startCall}
        stopCall={stopCall}
      />,
    );

    const button = screen.getByRole("button", {
      name: /stop voice conversation/i,
    });
    await user.click(button);

    expect(stopCall).toHaveBeenCalledOnce();
    expect(startCall).not.toHaveBeenCalled();
  });

  it("should disable both start and stop buttons when isDisabled is true", () => {
    const { rerender } = renderWithProviders(
      <VoiceButton
        voiceCallState="inactive"
        startCall={startCall}
        stopCall={stopCall}
        isDisabled={true}
      />,
    );

    let button = screen.getByRole("button", {
      name: /start voice conversation/i,
    });
    expect(button).toBeDisabled();

    rerender(
      <VoiceButton
        voiceCallState="active"
        startCall={startCall}
        stopCall={stopCall}
        isDisabled={true}
      />,
    );

    button = screen.getByRole("button", { name: /stop voice conversation/i });
    expect(button).toBeDisabled();
  });
});
