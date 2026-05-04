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
  agentId: "banking_backend",
  slotValue: null,
  stepId: "banking_0_call_banking_backend",
  timestamp: new Date("2026-01-21T14:28:54Z").toISOString(),
  originalTimestamp: 1737469734,
  metadata: {
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
    it("uses agentId from event", () => {
      renderWithProviders(
        <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("banking_backend")).toBeInTheDocument();
    });

    it("falls back to event.name when agentId is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        agentId: undefined,
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("event_name")).toBeInTheDocument();
    });

    it("falls back to 'agent' when both agentId and name are absent", () => {
      const event = {
        ...baseEvent,
        agentId: undefined,
        name: undefined,
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

  describe("tools section", () => {
    it("renders 'All server tools available' when neither tools nor excluded tools are set", () => {
      renderWithProviders(
        <AgentEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Tools")).toBeInTheDocument();
      expect(screen.getByText("All server tools available")).toBeInTheDocument();
    });

    it("renders 'All server tools available' when tools and excluded tools are empty arrays", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          ...baseEvent.metadata,
          mcp_tools: [],
          excluded_mcp_tools: [],
        },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("All server tools available")).toBeInTheDocument();
    });

    it("renders tool tags under 'Tools' heading when mcp_tools are present", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          ...baseEvent.metadata,
          mcp_tools: ["search_web", "send_email"],
        },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Tools")).toBeInTheDocument();
      expect(screen.getByText("search_web")).toBeInTheDocument();
      expect(screen.getByText("send_email")).toBeInTheDocument();
      expect(screen.queryByText("All server tools available")).not.toBeInTheDocument();
    });

    it("renders excluded tool tags under 'Excluded tools' heading when excluded_mcp_tools are present", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          ...baseEvent.metadata,
          excluded_mcp_tools: ["delete_record", "admin_reset"],
        },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Excluded tools")).toBeInTheDocument();
      expect(screen.getByText("delete_record")).toBeInTheDocument();
      expect(screen.getByText("admin_reset")).toBeInTheDocument();
      expect(screen.queryByText("All server tools available")).not.toBeInTheDocument();
    });

    it("renders both 'Tools' and 'Excluded tools' sections when both are present", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          ...baseEvent.metadata,
          mcp_tools: ["search_web"],
          excluded_mcp_tools: ["delete_record"],
        },
      };

      renderWithProviders(
        <AgentEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Tools")).toBeInTheDocument();
      expect(screen.getByText("search_web")).toBeInTheDocument();
      expect(screen.getByText("Excluded tools")).toBeInTheDocument();
      expect(screen.getByText("delete_record")).toBeInTheDocument();
      expect(screen.queryByText("All server tools available")).not.toBeInTheDocument();
    });
  });
});
