import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { generateTestCase, downloadConversation, downloadE2eTests } from "./download";
import {
  ConversationEventType,
  UtteranceType,
  type Conversation,
  type ConversationEvent,
  type EventMetadata,
  type RawEvent,
  type Utterance,
} from "../types";

function makeUtterance(overrides: Partial<Utterance>): Utterance {
  return {
    __typename: "Utterance",
    id: crypto.randomUUID(),
    text: "",
    type: UtteranceType.User,
    tokens: [],
    timestamp: new Date().toISOString(),
    originalTimestamp: 0,
    rephrase: false,
    rephrasePrompt: null,
    metadata: { parseData: {} } as EventMetadata,
    ...overrides,
  };
}

function makeConversationEvent(overrides: Partial<ConversationEvent>): ConversationEvent {
  return {
    __typename: "ConversationEvent",
    id: crypto.randomUUID(),
    conversationEventType: ConversationEventType.Action,
    actionText: "",
    flowId: "",
    name: "",
    slotValue: "",
    stepId: "",
    timestamp: new Date().toISOString(),
    originalTimestamp: 0,
    metadata: { parseData: {} } as EventMetadata,
    ...overrides,
  };
}

function makeConversation(events: (Utterance | ConversationEvent)[]): Conversation {
  return {
    id: "conv-1",
    startDate: new Date().toISOString(),
    reviewed: false,
    totalNumberOfUserMessages: events.filter(
      (e) => e.__typename === "Utterance" && e.type === UtteranceType.User,
    ).length,
    events,
  };
}

describe("generateTestCase", () => {
  it("generates user steps with bot_uttered assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "Hello" }),
      makeUtterance({
        type: UtteranceType.Bot,
        text: "Hi there!",
        metadata: { utter_action: "utter_greet", parseData: {} } as EventMetadata,
      }),
    ]);

    const result = generateTestCase("session-1", conversation);

    expect(result).toContain("test_case: session-1");
    expect(result).toContain('- user: "Hello"');
    expect(result).toContain("assertions:");
    expect(result).toContain('utter_name: "utter_greet"');
  });

  it("uses text_matches when no utter_action is available", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "Hi" }),
      makeUtterance({ type: UtteranceType.Bot, text: "Hello!" }),
    ]);

    const result = generateTestCase("session-2", conversation);

    expect(result).toContain('text_matches: "Hello!"');
    expect(result).not.toContain("utter_name");
  });

  it("generates flow_started assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "Transfer money" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "transfer_money",
      }),
    ]);

    const result = generateTestCase("session-3", conversation);

    expect(result).toContain('flow_started: "transfer_money"');
  });

  it("generates flow_completed assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "done" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "transfer_money",
      }),
    ]);

    const result = generateTestCase("session-4", conversation);

    expect(result).toContain("flow_completed:");
    expect(result).toContain('flow_id: "transfer_money"');
  });

  it("generates flow_cancelled assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "cancel" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowCancelled,
        flowId: "transfer_money",
      }),
    ]);

    const result = generateTestCase("session-5", conversation);

    expect(result).toContain("flow_cancelled:");
    expect(result).toContain('flow_id: "transfer_money"');
  });

  it("generates slot_was_set assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "100 dollars" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.Slot,
        name: "amount",
        slotValue: "100",
      }),
    ]);

    const result = generateTestCase("session-6", conversation);

    expect(result).toContain("slot_was_set:");
    expect(result).toContain('name: "amount"');
    expect(result).toContain('value: "100"');
  });

  it("skips internal rasa slots", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "hi" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.Slot,
        name: "flow_hashes",
        slotValue: "abc",
      }),
    ]);

    const result = generateTestCase("session-7", conversation);

    expect(result).not.toContain("flow_hashes");
    expect(result).not.toContain("slot_was_set");
  });

  it("generates action_executed assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "check balance" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.Action,
        actionText: "action_check_balance",
      }),
    ]);

    const result = generateTestCase("session-8", conversation);

    expect(result).toContain('action_executed: "action_check_balance"');
  });

  it("includes buttons in bot_uttered assertions", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "options" }),
      makeUtterance({
        type: UtteranceType.Bot,
        text: "Choose one:",
        responseData: {
          buttons: [],
          quickReplies: [
            { title: "Option A", payload: "/option_a" },
            { title: "Option B", payload: "/option_b" },
          ],
        },
      }),
    ]);

    const result = generateTestCase("session-9", conversation);

    expect(result).toContain("buttons:");
    expect(result).toContain('title: "Option A"');
    expect(result).toContain('payload: "/option_a"');
    expect(result).toContain('title: "Option B"');
  });

  it("ignores events before first user message", () => {
    const conversation = makeConversation([
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "pattern_session_start",
      }),
      makeUtterance({ type: UtteranceType.User, text: "Hello" }),
      makeUtterance({ type: UtteranceType.Bot, text: "Hi!" }),
    ]);

    const result = generateTestCase("session-10", conversation);

    expect(result).not.toContain("pattern_session_start");
    expect(result).toContain('- user: "Hello"');
  });

  it("produces correct YAML indentation matching canonical format", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "hey" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "hello",
      }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.Action,
        actionText: "utter_hello",
      }),
      makeUtterance({
        type: UtteranceType.Bot,
        text: "Hi!",
        metadata: { utter_action: "utter_hello", parseData: {} } as EventMetadata,
      }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "hello",
      }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.Slot,
        name: "continue_conversation",
        slotValue: null,
      }),
      makeUtterance({ type: UtteranceType.User, text: "money transfer" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowCompleted,
        flowId: "pattern_completed",
      }),
      makeUtterance({
        type: UtteranceType.Bot,
        text: "Choose transfer type",
        metadata: { utter_action: "utter_ask_transfer_type", parseData: {} } as EventMetadata,
        responseData: {
          buttons: [],
          quickReplies: [
            { title: "Send to someone else (domestic)", payload: '/SetSlots(transfer_type=third party)' },
            { title: "Transfer between your own accounts", payload: "/SetSlots(transfer_type=self transfer)" },
          ],
        },
      }),
    ]);

    const result = generateTestCase("test-session", conversation);
    const lines = result.split("\n");

    expect(lines[0]).toBe("test_cases:");
    expect(lines[1]).toBe("  - test_case: test-session");
    expect(lines[2]).toBe("    steps:");
    expect(lines[3]).toBe('      - user: "hey"');
    expect(lines[4]).toBe("        assertions:");
    expect(lines[5]).toBe('          - flow_started: "hello"');
    expect(lines[6]).toBe('          - action_executed: "utter_hello"');
    expect(lines[7]).toBe("          - bot_uttered:");
    expect(lines[8]).toBe('              utter_name: "utter_hello"');
    expect(lines[9]).toBe("          - flow_completed:");
    expect(lines[10]).toBe('                flow_id: "hello"');
    expect(lines[11]).toBe("          - slot_was_set:");
    expect(lines[12]).toBe('              - name: "continue_conversation"');
    expect(lines[13]).toBe("                value: null");
    expect(lines[14]).toBe('      - user: "money transfer"');
    expect(lines[15]).toBe("        assertions:");
    expect(lines[16]).toBe("          - flow_completed:");
    expect(lines[17]).toBe('                flow_id: "pattern_completed"');
    expect(lines[18]).toBe("          - bot_uttered:");
    expect(lines[19]).toBe('              utter_name: "utter_ask_transfer_type"');
    expect(lines[20]).toBe("              buttons:");
    expect(lines[21]).toBe('                - title: "Send to someone else (domestic)"');
    expect(lines[22]).toBe('                  payload: "/SetSlots(transfer_type=third party)"');
    expect(lines[23]).toBe('                - title: "Transfer between your own accounts"');
    expect(lines[24]).toBe('                  payload: "/SetSlots(transfer_type=self transfer)"');
  });

  it("groups multiple assertions under the same user step", () => {
    const conversation = makeConversation([
      makeUtterance({ type: UtteranceType.User, text: "transfer 50" }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.FlowStarted,
        flowId: "transfer_money",
      }),
      makeConversationEvent({
        conversationEventType: ConversationEventType.Slot,
        name: "amount",
        slotValue: "50",
      }),
      makeUtterance({
        type: UtteranceType.Bot,
        text: "To whom?",
        metadata: { utter_action: "utter_ask_recipient", parseData: {} } as EventMetadata,
      }),
    ]);

    const result = generateTestCase("session-11", conversation);

    const lines = result.split("\n");
    const userLine = lines.findIndex((l) => l.includes('user: "transfer 50"'));
    const assertionsLine = lines.findIndex((l) => l.includes("assertions:"));
    expect(assertionsLine).toBeGreaterThan(userLine);
    expect(result).toContain('flow_started: "transfer_money"');
    expect(result).toContain('name: "amount"');
    expect(result).toContain('utter_name: "utter_ask_recipient"');
  });
});

describe("downloadConversation", () => {
  let createObjectURLMock: ReturnType<typeof vi.fn>;
  let revokeObjectURLMock: ReturnType<typeof vi.fn>;
  let clickMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    createObjectURLMock = vi.fn().mockReturnValue("blob:test");
    revokeObjectURLMock = vi.fn();
    clickMock = vi.fn();

    vi.stubGlobal("URL", {
      createObjectURL: createObjectURLMock,
      revokeObjectURL: revokeObjectURLMock,
    });

    vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      click: clickMock,
    } as unknown as HTMLAnchorElement);

    vi.spyOn(document.body, "appendChild").mockImplementation((node) => node);
    vi.spyOn(document.body, "removeChild").mockImplementation((node) => node);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("triggers download with JSON content", () => {
    const rawEvent: RawEvent = {
      name: "",
      metadata: {} as EventMetadata,
      timestamp: 1700000000,
      event: "user",
      conversation_id: "test-conv",
      text: "Hello",
      data: {} as RawEvent["data"],
      update: "",
      flow_id: "",
      step_id: "",
      value: "",
      parse_data: { intent_ranking: [] },
    };
    const conversations: Conversation[] = [
      makeConversation([
        makeUtterance({
          type: UtteranceType.User,
          text: "Hello",
          metadata: { rawEvent, parseData: {} } as EventMetadata,
        }),
      ]),
    ];

    downloadConversation(conversations, "sess-1");

    expect(createObjectURLMock).toHaveBeenCalledOnce();
    expect(clickMock).toHaveBeenCalledOnce();
    expect(revokeObjectURLMock).toHaveBeenCalledOnce();
  });
});

describe("downloadE2eTests", () => {
  let createObjectURLMock: ReturnType<typeof vi.fn>;
  let revokeObjectURLMock: ReturnType<typeof vi.fn>;
  let clickMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    createObjectURLMock = vi.fn().mockReturnValue("blob:test");
    revokeObjectURLMock = vi.fn();
    clickMock = vi.fn();

    vi.stubGlobal("URL", {
      createObjectURL: createObjectURLMock,
      revokeObjectURL: revokeObjectURLMock,
    });

    vi.spyOn(document, "createElement").mockReturnValue({
      href: "",
      download: "",
      click: clickMock,
    } as unknown as HTMLAnchorElement);

    vi.spyOn(document.body, "appendChild").mockImplementation((node) => node);
    vi.spyOn(document.body, "removeChild").mockImplementation((node) => node);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("triggers download with YAML content", () => {
    const conversations: Conversation[] = [
      makeConversation([
        makeUtterance({ type: UtteranceType.User, text: "Test" }),
        makeUtterance({ type: UtteranceType.Bot, text: "Response" }),
      ]),
    ];

    downloadE2eTests(conversations, "sess-2");

    expect(createObjectURLMock).toHaveBeenCalledOnce();
    expect(clickMock).toHaveBeenCalledOnce();
    expect(revokeObjectURLMock).toHaveBeenCalledOnce();
  });
});
