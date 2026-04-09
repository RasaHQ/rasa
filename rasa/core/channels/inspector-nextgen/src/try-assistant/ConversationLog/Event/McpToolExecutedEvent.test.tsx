import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/utils";
import { ConversationEventType, type ConversationEvent } from "../../../types";
import { McpToolExecutedEvent } from "./McpToolExecutedEvent";

const baseEvent: ConversationEvent = {
  __typename: "ConversationEvent",
  id: "1",
  conversationEventType: ConversationEventType.McpToolExecuted,
  name: "event_name",
  actionText: "",
  flowId: "",
  slotValue: null,
  stepId: "",
  timestamp: new Date().toISOString(),
  metadata: {
    tool_name: "my_tool",
    parseData: {},
  },
};

describe("McpToolExecutedEvent", () => {
  describe("tool name display", () => {
    it("uses tool_name from metadata", () => {
      renderWithProviders(
        <McpToolExecutedEvent event={baseEvent} isSelected={false} />,
      );

      expect(screen.getByText("my_tool")).toBeInTheDocument();
    });

    it("falls back to event.name when metadata.tool_name is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { parseData: {} },
      };

      renderWithProviders(
        <McpToolExecutedEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText("event_name")).toBeInTheDocument();
    });
  });

  describe("execution status", () => {
    it("shows 'executed' when tool_is_error is false", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, tool_is_error: false },
      };

      renderWithProviders(
        <McpToolExecutedEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText(/executed/)).toBeInTheDocument();
      expect(screen.queryByText(/failed/)).not.toBeInTheDocument();
    });

    it("shows 'executed' when tool_is_error is absent", () => {
      renderWithProviders(
        <McpToolExecutedEvent event={baseEvent} isSelected={false} />,
      );

      expect(screen.getByText(/executed/)).toBeInTheDocument();
    });

    it("shows 'failed' when tool_is_error is true", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, tool_is_error: true },
      };

      renderWithProviders(
        <McpToolExecutedEvent event={event} isSelected={false} />,
      );

      expect(screen.getByText(/failed/)).toBeInTheDocument();
      expect(screen.queryByText(/executed/)).not.toBeInTheDocument();
    });
  });

  describe("interaction", () => {
    it("calls onClick when clicked", () => {
      const onClick = vi.fn();

      renderWithProviders(
        <McpToolExecutedEvent
          event={baseEvent}
          isSelected={false}
          onClick={onClick}
        />,
      );

      fireEvent.click(screen.getByRole("button"));
      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it("calls onKeyDown when a key is pressed", () => {
      const onKeyDown = vi.fn();

      renderWithProviders(
        <McpToolExecutedEvent
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
        <McpToolExecutedEvent
          event={baseEvent}
          isSelected={false}
          conversationEventActions={actions}
        />,
      );

      expect(screen.queryByText("Copy")).not.toBeInTheDocument();
    });

    it("shows action button on hover", () => {
      renderWithProviders(
        <McpToolExecutedEvent
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
        <McpToolExecutedEvent
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
