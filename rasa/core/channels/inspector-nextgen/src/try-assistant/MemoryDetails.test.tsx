import { screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  extractSlotEventsForFlow,
  extractSlotEventsForSession,
  formatSlots,
  isSystemSlotEvent,
} from "../utils/conversation";
import { toggleSelectedElement } from "../store";
import type * as StoreModule from "../store";
import type { ConversationEvent, Stack } from "../types";
import { ConversationEventType } from "../types/conversation";

// Matches the return type of the real formatSlots utility
type FormattedSlot = { name: string; value: unknown; event: ConversationEvent };
import { renderWithProviders } from "../tests/utils";
import { SlotsDetails } from "./MemoryDetails";

vi.mock("../VerticalScroll", () => ({
  ScrollContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  ScrollContent: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("../utils/conversation", () => ({
  extractSlotEventsForFlow: vi.fn(),
  extractSlotEventsForSession: vi.fn(),
  formatSlots: vi.fn(),
  isSystemSlotEvent: vi.fn(),
}));

vi.mock("../store", async (importOriginal) => {
  const actual = await importOriginal<typeof StoreModule>();
  return { ...actual, toggleSelectedElement: vi.fn() };
});

function makeSlotEvent(name: string): ConversationEvent {
  return {
    __typename: "ConversationEvent",
    actionText: "",
    conversationEventType: ConversationEventType.Slot,
    flowId: "booking",
    id: `event-${name}`,
    metadata: { parseData: {} },
    name,
    slotValue: null,
    stepId: "s1",
    timestamp: "2020-01-01T00:00:00.000Z",
    originalTimestamp: 0,
  };
}

function makeSlot(name: string, value: unknown, event: ConversationEvent): FormattedSlot {
  return { name, value, event };
}

const baseStack: Stack = {
  frameId: "frame1",
  flowId: "booking",
  stepId: "step1",
  ended: false,
};

function renderSlotsDetails(stack: Stack[] = []) {
  return renderWithProviders(<SlotsDetails />, {
    initialStoreState: { slotRelatedEvents: [], stack },
  });
}

describe("SlotsDetails", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(extractSlotEventsForFlow).mockReturnValue([]);
    vi.mocked(extractSlotEventsForSession).mockReturnValue([]);
    vi.mocked(formatSlots).mockReturnValue([]);
    vi.mocked(isSystemSlotEvent).mockReturnValue(false);
  });

  it("renders no sections when there are no slots", () => {
    renderSlotsDetails();
    expect(screen.queryByText("Current flow")).not.toBeInTheDocument();
    expect(screen.queryByText("Session")).not.toBeInTheDocument();
    expect(screen.queryByText("System")).not.toBeInTheDocument();
  });

  describe("Current flow section", () => {
    it("renders when the current flow has slots with values", () => {
      const flowEvents: ConversationEvent[] = [makeSlotEvent("city")];
      vi.mocked(extractSlotEventsForFlow).mockReturnValue(flowEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === flowEvents) return [makeSlot("city", "Berlin", flowEvents[0])];
        return [];
      });

      renderSlotsDetails([baseStack]);

      expect(screen.getByTestId("Current flow-section")).toBeInTheDocument();
      expect(screen.getByText("city")).toBeInTheDocument();
      expect(screen.getByText('"Berlin"')).toBeInTheDocument();
    });

    it("does not render when all current flow slot values are null", () => {
      const flowEvents: ConversationEvent[] = [makeSlotEvent("city")];
      vi.mocked(extractSlotEventsForFlow).mockReturnValue(flowEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === flowEvents) return [makeSlot("city", null, flowEvents[0])];
        return [];
      });

      renderSlotsDetails([baseStack]);

      expect(screen.queryByTestId("Current flow-section")).not.toBeInTheDocument();
    });

    it("does not render when the stack is empty", () => {
      renderSlotsDetails([]);

      expect(vi.mocked(extractSlotEventsForFlow)).not.toHaveBeenCalled();
      expect(screen.queryByTestId("Current flow-section")).not.toBeInTheDocument();
    });

    it("passes the active flow id to extractSlotEventsForFlow", () => {
      renderSlotsDetails([{ ...baseStack, flowId: "order_pizza" }]);

      expect(vi.mocked(extractSlotEventsForFlow)).toHaveBeenCalledWith([], "order_pizza");
    });
  });

  describe("Session section", () => {
    it("renders when there are non-null, non-system session slots", () => {
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("user")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("user", "John", sessionEvents[0])];
        return [];
      });

      renderSlotsDetails();

      expect(screen.getByTestId("Session-section")).toBeInTheDocument();
      expect(screen.getByText("user")).toBeInTheDocument();
    });

    it("does not render when session slot values are null", () => {
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("user")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("user", null, sessionEvents[0])];
        return [];
      });

      renderSlotsDetails();

      expect(screen.queryByTestId("Session-section")).not.toBeInTheDocument();
    });

    it("excludes system slots from the session section", () => {
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("language")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("language", "en", sessionEvents[0])];
        return [];
      });
      vi.mocked(isSystemSlotEvent).mockImplementation((name) => name === "language");

      renderSlotsDetails();

      expect(screen.queryByTestId("Session-section")).not.toBeInTheDocument();
    });

    it("excludes slots already present in the current flow section", () => {
      const flowEvents: ConversationEvent[] = [makeSlotEvent("city")];
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("city")];
      vi.mocked(extractSlotEventsForFlow).mockReturnValue(flowEvents);
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === flowEvents) return [makeSlot("city", "Berlin", flowEvents[0])];
        if (events === sessionEvents) return [makeSlot("city", "Berlin", sessionEvents[0])];
        return [];
      });

      renderSlotsDetails([baseStack]);

      expect(screen.getByTestId("Current flow-section")).toBeInTheDocument();
      expect(screen.queryByTestId("Session-section")).not.toBeInTheDocument();
    });
  });

  describe("System section", () => {
    it("renders for system slots", () => {
      // formatSlots is called 3 times: allSlots, currentFlowSlots, sessionSlots
      vi.mocked(formatSlots)
        .mockReturnValueOnce([makeSlot("language", "en", makeSlotEvent("language"))]) // allSlots
        .mockReturnValueOnce([]) // currentFlowSlots
        .mockReturnValueOnce([]); // sessionSlots
      vi.mocked(isSystemSlotEvent).mockImplementation((name) => name === "language");

      renderSlotsDetails();

      expect(screen.getByTestId("System-section")).toBeInTheDocument();
      expect(screen.getByText("language")).toBeInTheDocument();
    });
  });

  describe("separators between sections", () => {
    it("does not render a separator with only one section", () => {
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("user")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("user", "John", sessionEvents[0])];
        return [];
      });

      renderSlotsDetails();

      expect(screen.queryByRole("separator")).not.toBeInTheDocument();
    });

    it("renders a separator between two sections", () => {
      const flowEvents: ConversationEvent[] = [makeSlotEvent("city")];
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("user")];
      vi.mocked(extractSlotEventsForFlow).mockReturnValue(flowEvents);
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === flowEvents) return [makeSlot("city", "Berlin", flowEvents[0])];
        if (events === sessionEvents) return [makeSlot("user", "John", sessionEvents[0])];
        return [];
      });

      renderSlotsDetails([baseStack]);

      expect(screen.getByRole("separator")).toBeInTheDocument();
    });
  });

  describe("SlotRow", () => {
    it("renders slot name and JSON-stringified value", () => {
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("city")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("city", "Berlin", sessionEvents[0])];
        return [];
      });

      renderSlotsDetails();

      expect(screen.getByTestId("slot-city")).toBeInTheDocument();
      expect(screen.getByText("city")).toBeInTheDocument();
      expect(screen.getByText('"Berlin"')).toBeInTheDocument();
    });

    it("clicking a slot row with an event calls toggleSelectedElement", async () => {
      const user = userEvent.setup();
      const slotEvent = makeSlotEvent("city");
      const sessionEvents = [slotEvent];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("city", "Berlin", slotEvent)];
        return [];
      });

      renderSlotsDetails();

      await user.click(screen.getByTestId("slot-city"));
      expect(vi.mocked(toggleSelectedElement)).toHaveBeenCalledWith(slotEvent);
    });

    it("clicking a slot row without an event does not call toggleSelectedElement", async () => {
      const user = userEvent.setup();
      const sessionEvents = [makeSlotEvent("city")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        // Deliberately omit event to test the slot-without-event branch
        if (events === sessionEvents) return [{ name: "city", value: "Berlin" }] as FormattedSlot[];
        return [];
      });

      renderSlotsDetails();

      await user.click(screen.getByTestId("slot-city"));
      expect(vi.mocked(toggleSelectedElement)).not.toHaveBeenCalled();
    });

    it("renders the active slot row when the stack has a collect field", () => {
      const sessionEvents: ConversationEvent[] = [makeSlotEvent("city")];
      vi.mocked(extractSlotEventsForSession).mockReturnValue(sessionEvents);
      vi.mocked(formatSlots).mockImplementation((events) => {
        if (events === sessionEvents) return [makeSlot("city", "Berlin", sessionEvents[0])];
        return [];
      });

      renderSlotsDetails([{ ...baseStack, collect: "city" }]);

      expect(screen.getByTestId("slot-city")).toBeInTheDocument();
    });
  });
});
