import { screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Utterance } from "../../types";
import { UtteranceType } from "../../types/conversation";
import { renderWithProviders } from "../../tests/utils";
import { UserMessageInfo } from "./UserMessageInfo";

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
    metadata: { parseData: {} },
    rephrase: false,
    rephrasePrompt: null,
    text: "Hello",
    timestamp: "2020-01-01T00:00:00.000Z",
    tokens: [],
    type: UtteranceType.User,
    originalTimestamp: 0,
    ...overrides,
  };
}

function renderUserMessageInfo(
  utterance: Utterance,
  onClose: () => void = () => undefined,
) {
  return renderWithProviders(
    <UserMessageInfo utterance={utterance} onClose={onClose} />,
  );
}

describe("UserMessageInfo", () => {
  it("uses 'User message details' as the panel title", () => {
    renderUserMessageInfo(baseUtterance());
    expect(
      screen.getByRole("heading", { name: /user message details/i }),
    ).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    renderUserMessageInfo(baseUtterance(), onClose);
    await user.click(screen.getByTestId("event-details-close"));
    expect(onClose).toHaveBeenCalledOnce();
  });

  describe("Predicted intents", () => {
    it("shows a dash when intents are absent", () => {
      renderUserMessageInfo(baseUtterance({ intents: undefined }));
      expect(
        screen.getByRole("heading", { name: /predicted intents/i }),
      ).toBeInTheDocument();
      expect(screen.getByText("-")).toBeInTheDocument();
    });

    it("shows a dash when intents array is empty", () => {
      renderUserMessageInfo(baseUtterance({ intents: [] }));
      expect(screen.getByText("-")).toBeInTheDocument();
    });

    it("renders intent names and confidence percentages", () => {
      renderUserMessageInfo(
        baseUtterance({
          intents: [
            { id: "i1", name: "greet", confidence: 0.95 },
            { id: "i2", name: "affirm", confidence: 0.5 },
          ],
        }),
      );
      expect(screen.getByText("greet")).toBeInTheDocument();
      expect(screen.getByText("95.0%")).toBeInTheDocument();
      expect(screen.getByText("affirm")).toBeInTheDocument();
      expect(screen.getByText("50.0%")).toBeInTheDocument();
    });

    it("only renders up to 3 intents", () => {
      renderUserMessageInfo(
        baseUtterance({
          intents: [
            { id: "i1", name: "one", confidence: 0.9 },
            { id: "i2", name: "two", confidence: 0.8 },
            { id: "i3", name: "three", confidence: 0.7 },
            { id: "i4", name: "four", confidence: 0.1 },
          ],
        }),
      );
      expect(screen.getByText("one")).toBeInTheDocument();
      expect(screen.getByText("two")).toBeInTheDocument();
      expect(screen.getByText("three")).toBeInTheDocument();
      expect(screen.queryByText("four")).not.toBeInTheDocument();
    });

    it("applies green palette for confidence > 0.7", () => {
      renderUserMessageInfo(
        baseUtterance({
          intents: [{ id: "i1", name: "greet", confidence: 0.8 }],
        }),
      );
      expect(screen.getByText("80.0%")).toBeInTheDocument();
    });

    it("applies yellow palette for confidence between 0.4 and 0.7", () => {
      renderUserMessageInfo(
        baseUtterance({
          intents: [{ id: "i1", name: "greet", confidence: 0.5 }],
        }),
      );
      expect(screen.getByText("50.0%")).toBeInTheDocument();
    });

    it("applies red palette for confidence <= 0.4", () => {
      renderUserMessageInfo(
        baseUtterance({
          intents: [{ id: "i1", name: "greet", confidence: 0.3 }],
        }),
      );
      expect(screen.getByText("30.0%")).toBeInTheDocument();
    });
  });

  describe("Predicted Commands", () => {
    it("does not render the commands section when commands is undefined", () => {
      renderUserMessageInfo(baseUtterance({ commands: undefined }));
      expect(
        screen.queryByRole("heading", { name: /predicted commands/i }),
      ).not.toBeInTheDocument();
    });

    it("renders the commands section when commands is present", () => {
      renderUserMessageInfo(
        baseUtterance({ commands: [{ type: "SetSlot", name: "city", value: "Berlin" }] }),
      );
      expect(
        screen.getByRole("heading", { name: /predicted commands/i }),
      ).toBeInTheDocument();
    });

    it("renders commands as formatted JSON", () => {
      const commands = [{ type: "SetSlot", name: "city", value: "Berlin" }];
      renderUserMessageInfo(baseUtterance({ commands }));
      expect(
        screen.getByText((_, el) =>
          el?.tagName === "CODE" &&
          (el.textContent ?? "").includes('"type": "SetSlot"'),
        ),
      ).toBeInTheDocument();
    });
  });

  it("renders the 'Event details' accordion section", () => {
    renderUserMessageInfo(baseUtterance());
    expect(screen.getByText("Event details")).toBeInTheDocument();
  });
});
