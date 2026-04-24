import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/utils";
import { ConversationEventType, type ConversationEvent } from "../../types";
import { AgentEventInfo } from "./AgentEventInfo";

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

const baseEvent: ConversationEvent = {
  __typename: "ConversationEvent",
  id: "test-id-123",
  conversationEventType: ConversationEventType.AgentStarted,
  name: "event_name",
  actionText: "",
  flowId: "banking",
  slotValue: null,
  stepId: "banking_0_call_banking_backend",
  timestamp: new Date("2026-01-21T14:28:54Z").toISOString(),
  metadata: {
    agent_id: "banking_backend",
    active_flow: "banking",
    step_id: "banking_0_call_banking_backend",
    parseData: {},
  },
};

describe("AgentEventInfo", () => {
  it("renders the title", () => {
    renderWithProviders(
      <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
    );

    expect(screen.getByText("Sub-agent event details")).toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", () => {
    const onClose = vi.fn();
    renderWithProviders(
      <AgentEventInfo event={baseEvent} onClose={onClose} />,
    );

    fireEvent.click(screen.getByTestId("event-details-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  describe("agent name", () => {
    it("uses agent_id from metadata", () => {
      renderWithProviders(
        <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("banking_backend")).toBeInTheDocument();
    });

    it("falls back to event.name when agent_id is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { parseData: {} },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("event_name")).toBeInTheDocument();
    });

    it("falls back to 'agent' when both agent_id and name are absent", () => {
      const event = {
        ...baseEvent,
        name: undefined,
        metadata: { parseData: {} },
      } as unknown as ConversationEvent;

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("agent")).toBeInTheDocument();
    });
  });

  describe("trigger section", () => {
    it("shows flow ID in trigger text", () => {
      renderWithProviders(
        <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Trigger")).toBeInTheDocument();
      expect(
        screen.getByText(/This sub-agent was triggered by flow banking/),
      ).toBeInTheDocument();
    });

    it("uses metadata.active_flow as fallback", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        flowId: "",
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Trigger")).toBeInTheDocument();
      expect(
        screen.getByText(/This sub-agent was triggered by flow banking/),
      ).toBeInTheDocument();
    });

    it("hides trigger section when flow is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        flowId: "",
        metadata: {
          ...baseEvent.metadata,
          active_flow: undefined,
        },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.queryByText("Trigger")).not.toBeInTheDocument();
    });
  });

  describe("description section", () => {
    it("renders description when present in metadata", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          ...baseEvent.metadata,
          description: "Handles banking queries",
        },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Description")).toBeInTheDocument();
      expect(
        screen.getByText("Handles banking queries"),
      ).toBeInTheDocument();
    });

    it("shows dash when description is absent", () => {
      renderWithProviders(
        <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Description")).toBeInTheDocument();
      expect(screen.getByText("-")).toBeInTheDocument();
    });
  });

  describe("event info accordion", () => {
    it("renders event info section", () => {
      renderWithProviders(
        <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Event info")).toBeInTheDocument();
    });
  });
});
