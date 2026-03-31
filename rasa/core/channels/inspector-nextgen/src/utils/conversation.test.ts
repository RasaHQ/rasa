import { describe, it, expect } from "vitest";
import {
  extractSlotEventsForFlow,
  extractSlotEventsForSession,
  formatSlots,
  getSlotRelatedEvents,
} from "./conversation";
import {
  type ConversationEvent,
  type UnionEventType,
  ConversationEventType,
} from "../types";

function slot(
  id: string,
  name: string,
  slotValue: null | string,
  timestamp = "100",
  metadata: Record<string, unknown> = {}
): ConversationEvent {
  return {
    id,
    conversationEventType: ConversationEventType.Slot,
    name,
    slotValue,
    timestamp,
    metadata: { parseData: undefined, ...metadata },
    actionText: "",
    flowId: "",
    stepId: "",
    __typename: "ConversationEvent",
  };
}

function flowStarted(
  id: string,
  flowId: string,
  timestamp = "100",
  metadata: Record<string, unknown> = {}
): ConversationEvent {
  return {
    id,
    conversationEventType: ConversationEventType.FlowStarted,
    flowId,
    timestamp,
    metadata: { flow_id: flowId, parseData: undefined, ...metadata },
    name: "",
    slotValue: null,
    actionText: "",
    stepId: "",
    __typename: "ConversationEvent",
  };
}

function flowCompleted(
  id: string,
  flowId: string,
  timestamp = "100",
  metadata: Record<string, unknown> = {}
): ConversationEvent {
  return {
    id,
    conversationEventType: ConversationEventType.FlowCompleted,
    flowId,
    timestamp,
    metadata: { flow_id: flowId, parseData: undefined, ...metadata },
    name: "",
    slotValue: null,
    actionText: "",
    stepId: "",
    __typename: "ConversationEvent",
  };
}

function flowCancelled(
  id: string,
  flowId: string,
  timestamp = "100",
  metadata: Record<string, unknown> = {}
): ConversationEvent {
  return {
    id,
    conversationEventType: ConversationEventType.FlowCancelled,
    flowId,
    timestamp,
    metadata: { flow_id: flowId, parseData: undefined, ...metadata },
    name: "",
    slotValue: null,
    actionText: "",
    stepId: "",
    __typename: "ConversationEvent",
  };
}

function sessionStarted(
  id: string,
  timestamp = "100",
  metadata: Record<string, unknown> = {}
): ConversationEvent {
  return {
    id,
    conversationEventType: ConversationEventType.SessionStarted,
    timestamp,
    metadata: { parseData: undefined, ...metadata },
    name: "",
    slotValue: null,
    actionText: "",
    flowId: "",
    stepId: "",
    __typename: "ConversationEvent",
  };
}

describe("extractSlotEventsForFlow", () => {
  it("returns empty array if the flow never started", () => {
    const events: ConversationEvent[] = [
      slot("1", "some_slot", "test", "100"),
      flowCompleted("2", "some_other_flow", "200"),
    ];

    const result = extractSlotEventsForFlow(events, "target_flow");
    expect(result).toEqual([]);
  });

  it("returns slot events between FLOW_STARTED and FLOW_COMPLETED for the specified flow", () => {
    const events: ConversationEvent[] = [
      slot("1", "some_slot_outside", "should not be included", "100"),
      flowStarted("2", "target_flow", "101"),
      slot("3", "inside_slot_1", "123", "102"),
      flowCompleted("4", "target_flow", "103"),
      slot("5", "some_slot_outside_2", "should not be included", "104"),
    ];

    const result = extractSlotEventsForFlow(events, "target_flow");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("3");
    expect(result[0].name).toBe("inside_slot_1");
  });

  it("returns slot events between FLOW_STARTED and FLOW_CANCELLED for the specified flow", () => {
    const events: ConversationEvent[] = [
      flowStarted("1", "target_flow", "200"),
      slot("2", "inside_slot_1", "first", "201"),
      slot("3", "inside_slot_2", "second", "202"),
      flowCancelled("4", "target_flow", "203"),
      slot("5", "outside_slot", "should not be included", "204"),
    ];

    const result = extractSlotEventsForFlow(events, "target_flow");
    expect(result).toHaveLength(2);
    expect(result.map((ev) => ev.id)).toEqual(["2", "3"]);
  });

  it("assumes the flow is ongoing if no FLOW_COMPLETED or FLOW_CANCELLED is found", () => {
    const events: ConversationEvent[] = [
      flowStarted("1", "target_flow", "300"),
      slot("2", "inside_slot_1", "ongoing 1", "301"),
      slot("3", "inside_slot_2", "ongoing 2", "302"),
    ];

    const result = extractSlotEventsForFlow(events, "target_flow");
    expect(result).toHaveLength(2);
    expect(result[0].slotValue).toBe("ongoing 1");
    expect(result[1].slotValue).toBe("ongoing 2");
  });

  it("extracts slots only for the given flow among nested flows", () => {
    const events: ConversationEvent[] = [
      // Flow A starts
      flowStarted("A1", "flowA", "400"),
      slot("A2", "slotA1", "belongs to flowA", "401"),

      // Flow B starts (stack is now [flowA, flowB])
      flowStarted("B1", "flowB", "402"),
      slot("B2", "slotB1", "belongs to flowB", "403"),

      // Flow B ends (stack reverts to [flowA])
      flowCompleted("B3", "flowB", "404"),

      // Slots now belong to the top (flowA) again
      slot("A3", "slotA2", "belongs to flowA again", "405"),

      // Flow A ends
      flowCancelled("A4", "flowA", "406"),
    ];

    const resultA = extractSlotEventsForFlow(events, "flowA");
    const resultB = extractSlotEventsForFlow(events, "flowB");

    // Flow A: slotA1, slotA2
    expect(resultA).toHaveLength(2);
    expect(resultA.map((x) => x.name)).toEqual(["slotA1", "slotA2"]);

    // Flow B: slotB1
    expect(resultB).toHaveLength(1);
    expect(resultB[0].name).toBe("slotB1");
  });
});

describe("extractSlotEventsForSession", () => {
  it("returns an empty array if no SESSION_STARTED event is found", () => {
    const events: ConversationEvent[] = [
      slot("1", "slotX", "test", "100"),
      flowStarted("2", "some_flow", "101", { flow_id: "some_flow" }),
    ];

    const result = extractSlotEventsForSession(events);
    expect(result).toEqual([]);
  });

  it("returns all slot events from the first SESSION_STARTED to the end", () => {
    const events: ConversationEvent[] = [
      slot("1", "slotX", "test", "90"),
      sessionStarted("2", "100"),
      flowStarted("3", "some_flow", "110", { flow_id: "some_flow" }),
      slot("4", "slotY", "123", "120"),
    ];

    const result = extractSlotEventsForSession(events);
    expect(result.map((e) => e.id)).toEqual(["4"]);
  });

  it("works if SESSION_STARTED is at the very beginning", () => {
    const events: ConversationEvent[] = [
      sessionStarted("1", "1"),
      slot("2", "slot_after_start", "hello", "2"),
    ];

    const result = extractSlotEventsForSession(events);
    expect(result.length).toBe(1);
    expect(result[0].id).toBe("2");
  });
});

describe("formatSlots", () => {
  it("returns an empty array if no slot events exist", () => {
    const result = formatSlots([]);
    expect(result).toEqual([]);
  });

  it("maps each slot event to { name, value } pairs", () => {
    const slotEvents = [
      slot("1", "slotA", "valueA", "10"),
      slot("2", "slotB", "42", "11"),
    ];
    const result = formatSlots(slotEvents);
    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "slotA", value: "valueA" }),
        expect.objectContaining({ name: "slotB", value: "42" }),
      ])
    );
  });

  it("overwrites previous slot values if the same slot name is repeated", () => {
    const slotEvents = [
      slot("1", "slotA", "100", "1"),
      slot("2", "slotB", "200", "2"),
      slot("3", "slotA", "999", "3"), // redefines slotA
    ];
    // final: slotA=999, slotB=200
    const result = formatSlots(slotEvents);
    expect(result).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "slotA", value: "999" }),
        expect.objectContaining({ name: "slotB", value: "200" }),
      ])
    );
  });
});

describe("getSlotRelatedEvents", () => {
  it("returns only conversation events that are slot-related or allowed flow/session events", () => {
    const events: UnionEventType[] = [
      slot("slot1", "slotName", "slotVal", "100"),
      flowStarted("flowStartA", "some_flow", "110"),
      flowStarted("flowStartPattern", "pattern_error_handling", "120"),
      flowCancelled("flowCancelPattern", "pattern_whatever", "130"),
      flowCancelled("flowCancelRegular", "some_flow", "140"),
      sessionStarted("sessionStarted", "150"),
      flowCompleted("flowCompleteRegular", "some_flow", "160"),
    ];

    const relatedEvents = getSlotRelatedEvents(events);

    // Expect to exclude any flow.* with "pattern_" in the flowId
    expect(relatedEvents.map((e) => e.id)).toEqual([
      "slot1",
      "flowStartA",
      "flowCancelRegular",
      "sessionStarted",
      "flowCompleteRegular",
    ]);
  });

  it("filters out pattern_* flows on started, completed, or cancelled", () => {
    const events: UnionEventType[] = [
      flowStarted("pstart1", "pattern_cancel_flow", "100"),
      flowCompleted("pcomplete1", "pattern_handle_error", "100"),
      flowCancelled("pcancel1", "pattern_something", "100"),
      flowStarted("regularFlowStart", "non_pattern_flow", "100"),
      slot("slot1", "regularSlot", "val", "100"),
    ];

    const relatedEvents = getSlotRelatedEvents(events);

    // Only "regularFlowStart" and slot "slot1" remain
    expect(relatedEvents.map((e) => e.id)).toEqual([
      "regularFlowStart",
      "slot1",
    ]);
  });
});
