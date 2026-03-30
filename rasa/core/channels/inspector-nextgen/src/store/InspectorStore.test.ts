import { describe, it, expect, vi } from "vitest";
import {
  createInspectorStore,
  initInspectorStore,
  inspectorStore,
} from "./InspectorStore";

describe("createInspectorStore", () => {
  it("creates a store with default state", () => {
    const store = createInspectorStore();
    const state = store.state;

    expect(state.sessionId).toBe("");
    expect(state.conversationList).toEqual([]);
    expect(state.stack).toEqual([]);
    expect(state.inputDisabled).toBe(true);
    expect(state.replayingConversation).toBe(false);
    expect(state.waitingForUserInput).toBe(false);
    expect(state.slots).toEqual([]);
    expect(state.slotRelatedEvents).toEqual([]);

    expect(state.inspectMode).toBe(false);
    expect(state.selectedElement).toBeUndefined();

    expect(state.flows).toEqual([]);
    expect(state.flowsLoading).toBe(false);
    expect(state.flowsError).toBeNull();

    expect(state.projectUrl).toBe("");
    expect(state.botDataEndpoint).toBe("");
    expect(state.conversationEventActions).toEqual([]);
    expect(state.voiceFeaturesEnabled).toBe(true);
  });

  it("default actions are safe to call (noops)", () => {
    const store = createInspectorStore();
    const { sendMessage, startNewConversation, replayConversation, setUrl } =
      store.state;

    expect(() => sendMessage("hello")).not.toThrow();
    expect(() => startNewConversation()).not.toThrow();
    expect(() => replayConversation([])).not.toThrow();
    expect(() => setUrl("http://example.com")).not.toThrow();
  });

  it("default async actions resolve without error", async () => {
    const store = createInspectorStore();
    await expect(store.state.startVoiceStreaming()).resolves.toBeUndefined();
    await expect(store.state.stopVoiceStreaming()).resolves.toBeUndefined();
  });

  it("merges partial initial state over defaults", () => {
    const store = createInspectorStore({
      sessionId: "test-session",
      projectUrl: "http://my-bot.test",
      inspectMode: true,
      voiceFeaturesEnabled: false,
    });

    expect(store.state.sessionId).toBe("test-session");
    expect(store.state.projectUrl).toBe("http://my-bot.test");
    expect(store.state.inspectMode).toBe(true);
    expect(store.state.voiceFeaturesEnabled).toBe(false);

    // untouched defaults
    expect(store.state.conversationList).toEqual([]);
    expect(store.state.inputDisabled).toBe(true);
    expect(store.state.flowsLoading).toBe(false);
  });

  it("allows overriding action defaults", () => {
    const customSend = vi.fn();
    const store = createInspectorStore({ sendMessage: customSend });

    store.state.sendMessage("hi");
    expect(customSend).toHaveBeenCalledWith("hi");
  });

  it("setState updates state and notifies listeners", () => {
    const store = createInspectorStore();
    const listener = vi.fn();
    store.subscribe(listener);

    store.setState((prev) => ({ ...prev, sessionId: "updated" }));

    expect(store.state.sessionId).toBe("updated");
    expect(listener).toHaveBeenCalled();
  });

  it("each call creates an independent store instance", () => {
    const storeA = createInspectorStore({ sessionId: "a" });
    const storeB = createInspectorStore({ sessionId: "b" });

    storeA.setState((prev) => ({ ...prev, projectUrl: "changed" }));

    expect(storeA.state.projectUrl).toBe("changed");
    expect(storeB.state.projectUrl).toBe("");
    expect(storeA.state.sessionId).toBe("a");
    expect(storeB.state.sessionId).toBe("b");
  });
});

describe("initInspectorStore", () => {
  it("replaces the singleton with a fresh store", () => {
    inspectorStore.setState((prev) => ({ ...prev, sessionId: "dirty" }));
    expect(inspectorStore.state.sessionId).toBe("dirty");

    initInspectorStore();
    expect(inspectorStore.state.sessionId).toBe("");
  });

  it("applies initial state to the new singleton", () => {
    initInspectorStore({ projectUrl: "http://fresh.test", inspectMode: true });

    expect(inspectorStore.state.projectUrl).toBe("http://fresh.test");
    expect(inspectorStore.state.inspectMode).toBe(true);
    expect(inspectorStore.state.inputDisabled).toBe(true);
  });

  it("returns the new store instance", () => {
    const returned = initInspectorStore({ sessionId: "returned" });
    expect(returned).toBe(inspectorStore);
    expect(returned.state.sessionId).toBe("returned");
  });
});
