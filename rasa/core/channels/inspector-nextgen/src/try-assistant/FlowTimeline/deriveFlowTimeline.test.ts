import { describe, it, expect } from "vitest";
import { deriveFlowTimeline } from "./deriveFlowTimeline";
import {
  type ConversationEvent,
  type UnionEventType,
  type EventMetadata,
  ConversationEventType,
} from "../../types";

function makeFlowEvent(
  overrides: Partial<ConversationEvent> & {
    conversationEventType: ConversationEventType;
    flowId: string;
    timestamp: string;
  },
): ConversationEvent {
  return {
    __typename: "ConversationEvent",
    id: crypto.randomUUID(),
    actionText: "",
    name: "",
    slotValue: null,
    stepId: "",
    metadata: {} as EventMetadata,
    ...overrides,
  };
}

describe("deriveFlowTimeline", () => {
  it("returns entries in reverse chronological order (newest first)", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_b",
        timestamp: "2024-01-01T12:05:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(2);
    expect(result[0].flowId).toBe("flow_b");
    expect(result[1].flowId).toBe("flow_a");
  });

  it("marks started flows as active", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].status).toBe("active");
    expect(result[0].endTime).toBeUndefined();
  });

  it("marks completed flows with endTime", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:45Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].status).toBe("completed");
    expect(result[0].endTime).toEqual(new Date("2024-01-01T12:00:45Z"));
  });

  it("marks interrupted flows without endTime", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowInterrupted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].status).toBe("interrupted");
    expect(result[0].endTime).toBeUndefined();
  });

  it("marks cancelled flows with endTime", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCancelled,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:30Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].status).toBe("cancelled");
    expect(result[0].endTime).toBeDefined();
  });

  it("handles interrupted → resumed transitions back to active", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowInterrupted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowResumed,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:02:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].status).toBe("active");
  });

  it("filters out internal rasa flows", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "pattern_collect_information",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "real_flow",
        timestamp: "2024-01-01T12:01:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].flowId).toBe("real_flow");
  });

  it("resolves flow names from provided map", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "payment_processing",
        timestamp: "2024-01-01T12:00:00Z",
      }),
    ];

    const flowNames = new Map([
      ["payment_processing", "Payment Processing"],
    ]);
    const result = deriveFlowTimeline(events, flowNames);

    expect(result[0].flowName).toBe("Payment Processing");
  });

  it("leaves flowName undefined when no name map provided", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "payment_processing",
        timestamp: "2024-01-01T12:00:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result[0].flowName).toBeUndefined();
  });

  it("handles multiple invocations of the same flow", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        id: "ev-1",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
      }),
      makeFlowEvent({
        id: "ev-2",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:02:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("ev-2");
    expect(result[0].status).toBe("active");
    expect(result[1].id).toBe("ev-1");
    expect(result[1].status).toBe("completed");
  });

  it("returns empty array for no events", () => {
    expect(deriveFlowTimeline([])).toEqual([]);
  });

  it("ignores non-ConversationEvent events (Utterances, StackEvents)", () => {
    const events: UnionEventType[] = [
      {
        __typename: "Utterance",
        id: "1",
        type: "USER",
        text: "hello",
        timestamp: "2024-01-01T12:00:00Z",
        tokens: [],
        rephrase: false,
        rephrasePrompt: null,
        metadata: {} as EventMetadata,
      } as UnionEventType,
    ];

    expect(deriveFlowTimeline(events)).toEqual([]);
  });

  it("ignores completion events without a matching start", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
    ];

    expect(deriveFlowTimeline(events)).toEqual([]);
  });

  it("correctly closes nested invocations of the same flow (LIFO)", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        id: "outer",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
      }),
      makeFlowEvent({
        id: "inner",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:02:00Z",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(2);
    const innerEntry = result.find((e) => e.id === "inner");
    const outerEntry = result.find((e) => e.id === "outer");
    expect(innerEntry?.status).toBe("completed");
    expect(outerEntry?.status).toBe("active");
  });

  it("preserves startTime from the original FlowStarted event", () => {
    const timestamp = "2024-06-15T09:30:00Z";
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result[0].startTime).toEqual(new Date(timestamp));
  });
});
