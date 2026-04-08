import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useIsLargeScreen } from "./useIsLargeScreen";
import { initInspectorStore } from "../store";

function mockMatchMedia(matches: boolean) {
  const listeners = new Set<() => void>();
  const mql = {
    matches,
    media: "",
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn((_: string, cb: () => void) => listeners.add(cb)),
    removeEventListener: vi.fn((_: string, cb: () => void) =>
      listeners.delete(cb),
    ),
    dispatchEvent: vi.fn(),
  };

  vi.spyOn(window, "matchMedia").mockImplementation(() => mql as MediaQueryList);
  return { mql, listeners };
}

describe("useIsLargeScreen", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("returns true when viewport matches the standalone breakpoint", () => {
    initInspectorStore({ isEmbedded: false });
    mockMatchMedia(true);

    const { result } = renderHook(() => useIsLargeScreen());
    expect(result.current).toBe(true);
    expect(window.matchMedia).toHaveBeenCalledWith("(min-width: 1536px)");
  });

  it("returns false when viewport is below the standalone breakpoint", () => {
    initInspectorStore({ isEmbedded: false });
    mockMatchMedia(false);

    const { result } = renderHook(() => useIsLargeScreen());
    expect(result.current).toBe(false);
  });

  it("uses the embedded breakpoint when isEmbedded is true", () => {
    initInspectorStore({ isEmbedded: true });
    mockMatchMedia(false);

    const { result } = renderHook(() => useIsLargeScreen());
    expect(result.current).toBe(false);
    expect(window.matchMedia).toHaveBeenCalledWith("(min-width: 1920px)");
  });

  it("subscribes and unsubscribes to matchMedia change events", () => {
    initInspectorStore({ isEmbedded: false });
    const { mql } = mockMatchMedia(false);

    const { unmount } = renderHook(() => useIsLargeScreen());

    expect(mql.addEventListener).toHaveBeenCalledWith(
      "change",
      expect.any(Function),
    );

    unmount();

    expect(mql.removeEventListener).toHaveBeenCalledWith(
      "change",
      expect.any(Function),
    );
  });
});
