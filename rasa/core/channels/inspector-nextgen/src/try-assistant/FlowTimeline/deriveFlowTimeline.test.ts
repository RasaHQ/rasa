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
    agentId?: string;
    timestamp: string;
    originalTimestamp: number;
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_b",
        timestamp: "2024-01-01T12:05:00Z",
        originalTimestamp: 1704110700,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(2);
    expect(result[0].flowId).toBe("flow_b");
    expect(result[1].flowId).toBe("flow_a");
  });

  it("sorts entries by exactStartTime (originalTimestamp) descending", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_b",
        timestamp: "2024-01-01T12:05:00Z",
        originalTimestamp: 1704110700,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_c",
        timestamp: "2024-01-01T12:03:00Z",
        originalTimestamp: 1704110580,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(3);
    expect(result[0].flowId).toBe("flow_b");
    expect(result[1].flowId).toBe("flow_c");
    expect(result[2].flowId).toBe("flow_a");
    expect(result[0].exactStartTime).toBeGreaterThan(result[1].exactStartTime);
    expect(result[1].exactStartTime).toBeGreaterThan(result[2].exactStartTime);
  });

  it("marks started flows as active", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].type).toBe("flow");
    expect(result[0].status).toBe("active");
    expect(result[0].endTime).toBeUndefined();
  });

  it("marks completed flows with endTime", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:45Z",
        originalTimestamp: 1704110445,
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowInterrupted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
        originalTimestamp: 1704110460,
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCancelled,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:30Z",
        originalTimestamp: 1704110430,
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowInterrupted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
        originalTimestamp: 1704110460,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowResumed,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:02:00Z",
        originalTimestamp: 1704110520,
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "real_flow",
        timestamp: "2024-01-01T12:01:00Z",
        originalTimestamp: 1704110460,
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
        originalTimestamp: 1704110400,
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
        originalTimestamp: 1704110400,
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
        originalTimestamp: 1704110460,
      }),
      makeFlowEvent({
        id: "ev-2",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:02:00Z",
        originalTimestamp: 1704110520,
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
        originalTimestamp: 1704110400,
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
        originalTimestamp: 1704110400,
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
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        id: "inner",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:01:00Z",
        originalTimestamp: 1704110460,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:02:00Z",
        originalTimestamp: 1704110520,
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
        originalTimestamp: 1718443800,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result[0].startTime).toEqual(new Date(timestamp));
  });

  it("adds agent lifecycle entries and closes them on completion", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        id: "agent-started",
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentCompleted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:10Z",
        originalTimestamp: 1704110410,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("agent-started");
    expect(result[0].type).toBe("agent");
    // agentId is the agent identifier; flowId preserves the originating flow
    expect(result[0].agentId).toBe("deepwiki_github");
    expect(result[0].flowId).toBe("public_github_repo_info");
    expect(result[0].status).toBe("completed");
    expect(result[0].endTime).toEqual(new Date("2024-01-01T12:00:10Z"));
  });

  it("handles agent interrupted → resumed transitions back to active", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentInterrupted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:05Z",
        originalTimestamp: 1704110405,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentResumed,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:06Z",
        originalTimestamp: 1704110406,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].type).toBe("agent");
    expect(result[0].status).toBe("active");
    expect(result[0].endTime).toBeUndefined();
  });

  it("ignores agent lifecycle events without agentId", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "public_github_repo_info",
        timestamp: "2024-01-01T12:00:00Z",
        agentId: undefined,
        originalTimestamp: 1704110400,
      }),
    ];

    expect(deriveFlowTimeline(events)).toEqual([]);
  });

  it("orders agent entry before flow entry when both share a flowId and agent started sub-millisecond later", () => {
    // Both events get the same ISO timestamp string (ms precision), but originalTimestamp
    // carries sub-ms precision. The old startTime sort couldn't distinguish them; exactStartTime can.
    const events: UnionEventType[] = [
      makeFlowEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "my_flow",
        timestamp: "2024-01-01T12:00:00.000Z",
        originalTimestamp: 1704110400.0000,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "my_flow",
        agentId: "my_agent",
        timestamp: "2024-01-01T12:00:00.000Z",
        originalTimestamp: 1704110400.0001,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(2);
    expect(result[0].type).toBe("agent");  // agent started later → first in desc order
    expect(result[1].type).toBe("flow");
  });

  it("does not create duplicate agent entries for repeated starts while open", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        id: "agent-start-1",
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        id: "agent-start-2",
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:01Z",
        originalTimestamp: 1704110401,
      }),
      makeFlowEvent({
        conversationEventType: ConversationEventType.AgentCompleted,
        flowId: "public_github_repo_info",
        agentId: "deepwiki_github",
        timestamp: "2024-01-01T12:00:05Z",
        originalTimestamp: 1704110405,
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("agent-start-1");
    expect(result[0].status).toBe("completed");
  });

  it("routes flow lifecycle to the flow invocation when an agent shares the same flowId", () => {
    const events: UnionEventType[] = [
      makeFlowEvent({
        id: "flow-start",
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:00Z",
        originalTimestamp: 1704110400,
      }),
      makeFlowEvent({
        id: "agent-start",
        conversationEventType: ConversationEventType.AgentStarted,
        flowId: "flow_a",
        timestamp: "2024-01-01T12:00:30Z",
        originalTimestamp: 1704110430,
        agentId: "my_agent",
      }),
    ];

    const result = deriveFlowTimeline(events);

    expect(result).toHaveLength(2);

    const [agentEntry, flowEntry] = result;
    expect(agentEntry.type).toEqual("agent");
    expect(flowEntry.type).toEqual("flow");

    expect(flowEntry?.status).toBe("active");
    expect(agentEntry?.status).toBe("active");
  });
});
