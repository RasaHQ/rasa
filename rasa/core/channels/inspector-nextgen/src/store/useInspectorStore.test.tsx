import { describe, it, expect, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";
import { initInspectorStore, inspectorStore } from "./InspectorStore";
import { useInspectorStore } from "./useInspectorStore";

beforeEach(() => {
  initInspectorStore();
});

describe("useInspectorStore", () => {
  it("selects a primitive value from state", () => {
    initInspectorStore({ sessionId: "abc-123" });
    const { result } = renderHook(() => useInspectorStore((s) => s.sessionId));
    expect(result.current).toBe("abc-123");
  });

  it("selects a derived value via a custom selector", () => {
    initInspectorStore({
      waitingForUserInput: false,
      inputDisabled: false,
    });
    const { result } = renderHook(() =>
      useInspectorStore((s) => !s.waitingForUserInput && !s.inputDisabled),
    );
    expect(result.current).toBe(true);
  });

  it("reflects store updates on re-render", () => {
    initInspectorStore({ projectUrl: "old" });
    const { result } = renderHook(() =>
      useInspectorStore((s) => s.projectUrl),
    );
    expect(result.current).toBe("old");

    act(() => {
      inspectorStore.setState((prev) => ({ ...prev, projectUrl: "new" }));
    });

    expect(result.current).toBe("new");
  });
});
