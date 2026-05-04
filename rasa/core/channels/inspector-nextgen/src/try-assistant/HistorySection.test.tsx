import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../tests/utils";
import {
  ConversationEventType,
  type Conversation,
  type ConversationEvent,
  type EventMetadata,
} from "../types";
import { HistorySection } from "./HistorySection";

vi.mock("./FlowTimeline", async (importOriginal) => {
  return {
    ...(await importOriginal()),
    FlowTimeline: ({
      entries,
      onEntryClick,
    }: {
      entries: Array<{ id: string }>;
      onEntryClick: (id: string) => void;
    }) => (
      <div
        data-testid="flow-timeline"
        data-entry-count={entries.length}
        onClick={() => onEntryClick(entries[0].id)}
      />
    ),
  };
});

vi.mock("../Placeholder", () => ({
  NoData: () => <div data-testid="no-data" />,
}));

vi.mock("../components/InspectorViewHeader", () => ({
  InspectorViewHeader: () => <div data-testid="inspector-view-header" />,
}));

vi.mock("../components/ScrollFadeArea", () => ({
  ScrollFadeArea: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("../store", async (importOriginal) => {
  return { ...(await importOriginal()), toggleSelectedElement: vi.fn() };
});

import { toggleSelectedElement } from "../store";

function makeConversation(events: Conversation["events"] = []): Conversation {
  return {
    id: "conv1",
    reviewed: false,
    startDate: new Date().toISOString(),
    totalNumberOfUserMessages: 0,
    events,
  };
}

function makeFlowStartedEvent(id: string, flowId = "my_flow"): ConversationEvent {
  return {
    __typename: "ConversationEvent",
    id,
    conversationEventType: ConversationEventType.FlowStarted,
    flowId,
    timestamp: "2024-01-01T12:00:00Z",
    originalTimestamp: 0,
    actionText: "",
    name: "",
    slotValue: null,
    stepId: "",
    metadata: {} as EventMetadata,
  };
}

describe("HistorySection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders FlowTimeline when history entries are present", () => {
    renderWithProviders(<HistorySection />, {
      initialStoreState: {
        conversationList: [makeConversation([makeFlowStartedEvent("ev-1")])],
      },
    });

    expect(screen.getByTestId("flow-timeline")).toBeInTheDocument();
    expect(screen.queryByTestId("no-data")).not.toBeInTheDocument();
  });

  it("renders NoData when no history entries are present", () => {
    renderWithProviders(<HistorySection />, {
      initialStoreState: {
        conversationList: [makeConversation([])],
      },
    });

    expect(screen.getByTestId("no-data")).toBeInTheDocument();
    expect(screen.queryByTestId("flow-timeline")).not.toBeInTheDocument();
  });

  it("calls toggleSelectedElement with the matching event when an entry is clicked", () => {
    const event = makeFlowStartedEvent("ev-1");

    renderWithProviders(<HistorySection />, {
      initialStoreState: {
        conversationList: [makeConversation([event])],
      },
    });

    screen.getByTestId("flow-timeline").click();
    expect(toggleSelectedElement).toHaveBeenCalledWith(event);
  });
});
