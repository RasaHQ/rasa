import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../tests/utils";
import { ConversationEventType, type ConversationEvent } from "../../types";
import { McpToolExecutedEventInfo } from "./McpToolExecutedEventInfo";

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
    tool_arguments: { query: "hello" },
    tool_result: "some result",
    tool_is_error: false,
    parseData: {},
  },
};

describe("McpToolExecutedEventInfo", () => {
  it("renders the title", () => {
    renderWithProviders(
      <McpToolExecutedEventInfo event={baseEvent} onClose={vi.fn()} />,
    );

    expect(screen.getByText("MCP tool executed")).toBeInTheDocument();
  });

  it("calls onClose when close button is clicked", () => {
    const onClose = vi.fn();
    renderWithProviders(
      <McpToolExecutedEventInfo event={baseEvent} onClose={onClose} />,
    );

    fireEvent.click(screen.getByTestId("event-details-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  describe("tool name", () => {
    it("uses metadata.tool_name", () => {
      renderWithProviders(
        <McpToolExecutedEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByText("my_tool")).toBeInTheDocument();
    });

    it("falls back to rawEvent.tool_name when metadata.tool_name is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          parseData: {},
          rawEvent: { tool_name: "raw_tool" } as never,
        },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("raw_tool")).toBeInTheDocument();
    });

    it("falls back to event.name when neither metadata nor rawEvent has tool_name", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { parseData: {} },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("event_name")).toBeInTheDocument();
    });
  });

  describe("flow section", () => {
    it("shows flow ID when event.flowId is set", () => {
      const event: ConversationEvent = { ...baseEvent, flowId: "my-flow" };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("my-flow")).toBeInTheDocument();
      expect(screen.getByText("Flow")).toBeInTheDocument();
    });

    it("hides flow section when event.flowId is empty", () => {
      renderWithProviders(
        <McpToolExecutedEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.queryByText("Flow")).not.toBeInTheDocument();
    });
  });

  describe("arguments", () => {
    it("renders arguments as JSON", () => {
      renderWithProviders(
        <McpToolExecutedEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      const el = screen.getByTestId("event-mcp-tool-arguments");
      expect(el).toHaveTextContent(JSON.stringify({ query: "hello" }, null, 2), {
        normalizeWhitespace: false,
      });
    });

    it("shows '—' when arguments are null", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, tool_arguments: undefined },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByTestId("event-mcp-tool-arguments")).toHaveTextContent("—");
    });

    it("uses rawEvent.arguments as fallback", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          parseData: {},
          rawEvent: { arguments: { key: "value" } } as never,
        },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByTestId("event-mcp-tool-arguments")).toHaveTextContent(
        JSON.stringify({ key: "value" }, null, 2),
        { normalizeWhitespace: false },
      );
    });
  });

  describe("result section (no error)", () => {
    it("shows result as string", () => {
      renderWithProviders(
        <McpToolExecutedEventInfo event={baseEvent} onClose={vi.fn()} />,
      );

      expect(screen.getByTestId("event-mcp-tool-result")).toHaveTextContent("some result");
    });

    it("renders object result as JSON", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, tool_result: { data: 42 } },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByTestId("event-mcp-tool-result")).toHaveTextContent(
        JSON.stringify({ data: 42 }, null, 2),
        { normalizeWhitespace: false },
      );
    });

    it("shows '—' when result is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, tool_result: undefined },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByTestId("event-mcp-tool-result")).toHaveTextContent("—");
    });
  });

  describe("error section", () => {
    it("shows error section instead of result when tool_is_error is true", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          ...baseEvent.metadata,
          tool_is_error: true,
          tool_error_message: "something went wrong",
        },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Error")).toBeInTheDocument();
      expect(screen.getByText("something went wrong")).toBeInTheDocument();
      expect(screen.queryByTestId("event-mcp-tool-result")).not.toBeInTheDocument();
    });

    it("shows '—' when error message is absent", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: { ...baseEvent.metadata, tool_is_error: true },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Error")).toBeInTheDocument();
      expect(screen.getAllByText("—").length).toBeGreaterThanOrEqual(1);
    });

    it("shows error section when rawEvent.is_error is true", () => {
      const event: ConversationEvent = {
        ...baseEvent,
        metadata: {
          parseData: {},
          rawEvent: { is_error: true, error_message: "raw error" } as never,
        },
      };

      renderWithProviders(
        <McpToolExecutedEventInfo event={event} onClose={vi.fn()} />,
      );

      expect(screen.getByText("Error")).toBeInTheDocument();
      expect(screen.getByText("raw error")).toBeInTheDocument();
    });
  });
});
