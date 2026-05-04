import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/utils";
import { ConversationEventType, type ConversationEvent } from "../../../types";
import { AgentEvent } from "./AgentEvent";

const baseEvent: ConversationEvent = {
  __typename: "ConversationEvent",
  id: "1",
  conversationEventType: ConversationEventType.AgentStarted,
  name: "event_name",
  actionText: "",
  flowId: "",
  slotValue: null,
  stepId: "",
  timestamp: new Date().toISOString(),
  originalTimestamp: 0,
  agentId: "banking_backend",
  metadata: {
    parseData: {},
  },
};

describe("AgentEvent", () => {
  describe("agent name display", () => {
    it("uses agentId from event", () => {
      renderWithProviders(
        <AgentEvent event={baseEvent} isSelected={false} />,
      );

      expect(screen.getByText("banking_backend")).toBeInTheDocument();
    });

    it("falls back to event.name when agentId is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        agentId: undefined,
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
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
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText("agent")).toBeInTheDocument();
    });
  });

  describe("status text per event type", () => {
    it("shows 'invoked' for AgentStarted", () => {
      renderWithProviders(
        <AgentEvent event={baseEvent} isSelected={false} />,
      );

      expect(screen.getByText(/invoked/)).toBeInTheDocument();
    });

    it("shows 'stopped' for AgentCompleted", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        conversationEventType: ConversationEventType.AgentCompleted,
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText(/stopped/)).toBeInTheDocument();
    });

    it("shows 'stopped' for AgentCancelled", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        conversationEventType: ConversationEventType.AgentCancelled,
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText(/stopped/)).toBeInTheDocument();
    });

    it("shows 'interrupted' for AgentInterrupted", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        conversationEventType: ConversationEventType.AgentInterrupted,
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText(/interrupted/)).toBeInTheDocument();
    });

    it("shows 'resumed' for AgentResumed", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        conversationEventType: ConversationEventType.AgentResumed,
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText(/resumed/)).toBeInTheDocument();
    });
  });

  describe("error dot", () => {
    it("shows error dot when execution_success is false", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, execution_success: false },
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.getByTestId("error-dot")).toBeInTheDocument();
    });

    it("does not show error dot when execution_success is true", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, execution_success: true },
      };

      renderWithProviders(
        <AgentEvent event={event} isSelected={false} />,
      );

      expect(screen.queryByTestId("error-dot")).not.toBeInTheDocument();
    });

    it("does not show error dot when execution_success is absent", () => {
      renderWithProviders(
        <AgentEvent event={baseEvent} isSelected={false} />,
      );

      expect(screen.queryByTestId("error-dot")).not.toBeInTheDocument();
    });
  });

  describe("interaction", () => {
    it("calls onClick when clicked", () => {
      const onClick = vi.fn();

      renderWithProviders(
        <AgentEvent event={baseEvent} isSelected={false} onClick={onClick} />,
      );

      fireEvent.click(screen.getByRole("button"));
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("calls onKeyDown when a key is pressed", () => {
      const onKeyDown = vi.fn();

      renderWithProviders(
        <AgentEvent
          event={baseEvent}
          isSelected={false}
          onKeyDown={onKeyDown}
        />,
      );

      fireEvent.keyDown(screen.getByRole("button"), { key: "Enter" });
      expect(onKeyDown).toHaveBeenCalledTimes(1);
    });
  });

  describe("hover actions", () => {
    const actions = [{ icon: {} as never, label: "Copy", action: vi.fn() }];

    it("does not show action button when not hovered", () => {
      renderWithProviders(
        <AgentEvent
          event={baseEvent}
          isSelected={false}
          conversationEventActions={actions}
        />,
      );

      expect(screen.queryByText("Copy")).not.toBeInTheDocument();
    });

    it("shows action button on hover", () => {
      renderWithProviders(
        <AgentEvent
          event={baseEvent}
          isSelected={false}
          conversationEventActions={actions}
        />,
      );

      fireEvent.mouseEnter(screen.getByRole("button"));
      expect(screen.getByText("Copy")).toBeInTheDocument();
    });

    it("hides action button after mouse leaves", () => {
      renderWithProviders(
        <AgentEvent
          event={baseEvent}
          isSelected={false}
          conversationEventActions={actions}
        />,
      );

      const container = screen.getByRole("button");
      fireEvent.mouseEnter(container);
      fireEvent.mouseLeave(container);
      expect(screen.queryByText("Copy")).not.toBeInTheDocument();
    });
  });
});
