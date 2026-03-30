import { describe, it, expect, vi } from "vitest";
import type { InspectorStoreState } from "./InspectorStore";
import type { Conversation, ConversationEvent, Stack } from "../types";
import { InspectorView } from "../types/inspector";
import {
  selectLatestStack,
  selectStackToShow,
  selectConversationForSelectedElement,
  selectWaitingForResponse,
} from "./selectors";

const noop = vi.fn();
const asyncNoop = vi.fn().mockResolvedValue(undefined);

function makeState(
  overrides: Partial<InspectorStoreState> = {},
): InspectorStoreState {
  return {
    sessionId: "",
    conversationList: [],
    stack: [],
    inputDisabled: true,
    replayingConversation: false,
    waitingForUserInput: false,
    slots: [],
    slotRelatedEvents: [],
    inspectMode: false,
    inspectorView: InspectorView.ActiveFlow,
    selectedElement: undefined,
    flows: [],
    flowsLoading: false,
    flowsError: null,
    projectUrl: "",
    botDataEndpoint: "",
    conversationEventActions: [],
    voiceFeaturesEnabled: true,
    sendMessage: noop,
    startNewConversation: noop,
    replayConversation: noop,
    setUrl: noop,
    startVoiceStreaming: asyncNoop,
    stopVoiceStreaming: asyncNoop,
    onVoiceErrorRef: { current: null },
    ...overrides,
  };
}

function makeStack(overrides: Partial<Stack> = {}): Stack {
  return {
    frameId: "frame-1",
    flowId: "my_flow",
    stepId: "step_1",
    ended: false,
    ...overrides,
  };
}

function makeEvent(
  overrides: Partial<ConversationEvent> = {},
): ConversationEvent {
  return {
    __typename: "ConversationEvent",
    actionText: "",
    conversationEventType:
      "ACTION" as ConversationEvent["conversationEventType"],
    flowId: "",
    id: overrides.id ?? "evt-1",
    metadata: overrides.metadata ?? { parseData: undefined },
    name: "",
    slotValue: null,
    stepId: "",
    timestamp: new Date().toISOString(),
    ...overrides,
  } as ConversationEvent;
}

function makeConversation(overrides: Partial<Conversation> = {}): Conversation {
  return {
    events: [],
    id: "conv-1",
    reviewed: false,
    startDate: new Date().toISOString(),
    totalNumberOfUserMessages: 0,
    ...overrides,
  };
}

// ---------- selectLatestStack ----------

describe("selectLatestStack", () => {
  it("returns undefined for empty stack", () => {
    expect(selectLatestStack(makeState())).toBeUndefined();
  });

  it("returns the last user-visible frame", () => {
    const stack = [
      makeStack({ frameId: "f1", flowId: "greeting" }),
      makeStack({ frameId: "f2", flowId: "farewell" }),
    ];
    const result = selectLatestStack(makeState({ stack }));
    expect(result?.frameId).toBe("f2");
  });

  it("skips internal pattern_ frames", () => {
    const stack = [
      makeStack({ frameId: "f1", flowId: "greeting" }),
      makeStack({ frameId: "f2", flowId: "pattern_internal" }),
    ];
    const result = selectLatestStack(makeState({ stack }));
    expect(result?.frameId).toBe("f1");
  });

  it("allows pattern_session_start", () => {
    const stack = [
      makeStack({ frameId: "f1", flowId: "pattern_session_start" }),
    ];
    expect(selectLatestStack(makeState({ stack }))?.frameId).toBe("f1");
  });

  it("allows pattern_completed", () => {
    const stack = [makeStack({ frameId: "f1", flowId: "pattern_completed" })];
    expect(selectLatestStack(makeState({ stack }))?.frameId).toBe("f1");
  });

  it("returns undefined when all frames are hidden patterns", () => {
    const stack = [
      makeStack({ flowId: "pattern_collect_information" }),
      makeStack({ flowId: "pattern_correction" }),
    ];
    expect(selectLatestStack(makeState({ stack }))).toBeUndefined();
  });

  it("handles frames with undefined flowId", () => {
    const stack = [makeStack({ flowId: undefined as unknown as string })];
    const result = selectLatestStack(makeState({ stack }));
    expect(result).toBeDefined();
  });
});

// ---------- selectStackToShow ----------

describe("selectStackToShow", () => {
  it("falls back to latestStack when no element is selected", () => {
    const stack = [makeStack({ frameId: "f1", flowId: "my_flow" })];
    const result = selectStackToShow(makeState({ stack }));
    expect(result?.frameId).toBe("f1");
  });

  it("returns selectedElement-based stack when element has flow metadata", () => {
    const selected = makeEvent({
      id: "evt-42",
      metadata: {
        flow_id: "transfer",
        step_id: "ask_amount",
        parseData: undefined,
      },
    });
    const result = selectStackToShow(
      makeState({ selectedElement: selected, stack: [] }),
    );
    expect(result).toEqual({
      frameId: "evt-42",
      flowId: "transfer",
      stepId: "ask_amount",
      ended: false,
    });
  });

  it("falls back to latestStack when element has no flow_id", () => {
    const selected = makeEvent({
      id: "evt-1",
      metadata: { step_id: "s", parseData: undefined },
    });
    const stack = [makeStack({ frameId: "f1" })];
    const result = selectStackToShow(
      makeState({ selectedElement: selected, stack }),
    );
    expect(result?.frameId).toBe("f1");
  });

  it("falls back to latestStack when element has no step_id", () => {
    const selected = makeEvent({
      id: "evt-1",
      metadata: { flow_id: "f", parseData: undefined },
    });
    const stack = [makeStack({ frameId: "f1" })];
    const result = selectStackToShow(
      makeState({ selectedElement: selected, stack }),
    );
    expect(result?.frameId).toBe("f1");
  });

  it("returns undefined when no element and empty stack", () => {
    expect(selectStackToShow(makeState())).toBeUndefined();
  });
});

// ---------- selectConversationForSelectedElement ----------

describe("selectConversationForSelectedElement", () => {
  it("returns undefined when no element is selected", () => {
    expect(selectConversationForSelectedElement(makeState())).toBeUndefined();
  });

  it("returns undefined when element id is not found in any conversation", () => {
    const selected = makeEvent({ id: "missing" });
    const conversations = [
      makeConversation({
        events: [makeEvent({ id: "evt-1" }), makeEvent({ id: "evt-2" })],
      }),
    ];
    expect(
      selectConversationForSelectedElement(
        makeState({
          selectedElement: selected,
          conversationList: conversations,
        }),
      ),
    ).toBeUndefined();
  });

  it("returns the conversation containing the selected element", () => {
    const selected = makeEvent({ id: "evt-2" });
    const target = makeConversation({
      id: "conv-match",
      events: [makeEvent({ id: "evt-2" }), makeEvent({ id: "evt-3" })],
    });
    const other = makeConversation({
      id: "conv-other",
      events: [makeEvent({ id: "evt-99" })],
    });
    const result = selectConversationForSelectedElement(
      makeState({
        selectedElement: selected,
        conversationList: [other, target],
      }),
    );
    expect(result?.id).toBe("conv-match");
  });

  it("returns the first matching conversation when duplicates exist", () => {
    const selected = makeEvent({ id: "evt-shared" });
    const first = makeConversation({
      id: "conv-first",
      events: [makeEvent({ id: "evt-shared" })],
    });
    const second = makeConversation({
      id: "conv-second",
      events: [makeEvent({ id: "evt-shared" })],
    });
    const result = selectConversationForSelectedElement(
      makeState({
        selectedElement: selected,
        conversationList: [first, second],
      }),
    );
    expect(result?.id).toBe("conv-first");
  });
});

// ---------- selectWaitingForResponse ----------

describe("selectWaitingForResponse", () => {
  it("returns true when not waiting for user input and not input disabled", () => {
    expect(
      selectWaitingForResponse(
        makeState({ waitingForUserInput: false, inputDisabled: false }),
      ),
    ).toBe(true);
  });

  it("returns false when waiting for user input", () => {
    expect(
      selectWaitingForResponse(
        makeState({ waitingForUserInput: true, inputDisabled: false }),
      ),
    ).toBe(false);
  });

  it("returns false when input is disabled", () => {
    expect(
      selectWaitingForResponse(
        makeState({ waitingForUserInput: false, inputDisabled: true }),
      ),
    ).toBe(false);
  });

  it("returns false when both waiting for user input and input disabled", () => {
    expect(
      selectWaitingForResponse(
        makeState({ waitingForUserInput: true, inputDisabled: true }),
      ),
    ).toBe(false);
  });
});
