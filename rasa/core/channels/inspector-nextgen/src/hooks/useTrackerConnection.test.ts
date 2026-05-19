import { renderHook, act } from "@testing-library/react";
import { MemoryRouter, useSearchParams } from "react-router-dom";
import React from "react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
  type Mock,
} from "vitest";
import { useTrackerConnection } from "./useTrackerConnection";
import { initInspectorStore, inspectorStore } from "../store";

type MockWebSocket = {
  onopen?: () => void;
  onmessage?: (event: { data: string }) => void;
  send: Mock;
  close: Mock;
  url: string;
};

let lastWs: MockWebSocket | undefined;
let wsCtor: Mock;

beforeEach(() => {
  vi.clearAllMocks();
  initInspectorStore();
  lastWs = undefined;
  wsCtor = vi.fn().mockImplementation((url: string) => {
    const ws: MockWebSocket = {
      send: vi.fn(),
      close: vi.fn(),
      url,
    };
    lastWs = ws;
    return ws;
  });
  vi.stubGlobal("WebSocket", wsCtor);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderTrackerHook(
  { enabled, channel }: { enabled: boolean; channel: string },
  initialEntries: string[] = ["/"],
) {
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(MemoryRouter, { initialEntries }, children);
  return renderHook(() => useTrackerConnection({ enabled, channel }), {
    wrapper,
  });
}

function renderWithSearchParamsProbe(initialEntries: string[]) {
  const capturedSearch: { current: string | null } = { current: null };
  const SearchParamsProbe = () => {
    const [searchParams] = useSearchParams();
    React.useEffect(() => {
      capturedSearch.current = searchParams.get("sender");
    }, [searchParams]);
    return null;
  };

  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(MemoryRouter, { initialEntries }, [
      React.createElement(SearchParamsProbe, { key: "probe" }),
      children,
    ]);

  const utils = renderHook(
    () => useTrackerConnection({ enabled: true, channel: "rest" }),
    { wrapper },
  );

  return { capturedSearch, ...utils };
}

describe("useTrackerConnection", () => {
  describe("guards", () => {
    it("does not open a WebSocket when enabled is false", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: false, channel: "rest" });

      expect(wsCtor).not.toHaveBeenCalled();
    });

    it("does not open a WebSocket when projectUrl is empty", () => {
      renderTrackerHook({ enabled: true, channel: "rest" });

      expect(wsCtor).not.toHaveBeenCalled();
    });
  });

  describe("WebSocket URL construction", () => {
    it("constructs the ws URL from projectUrl and channel", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" });

      expect(wsCtor).toHaveBeenCalledWith(
        "ws://localhost:5005/webhooks/rest/tracker_stream",
      );
    });

    it("uses wss for https projectUrl", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "https://example.com",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "twilio_media_streams" });

      expect(wsCtor).toHaveBeenCalledWith(
        "wss://example.com/webhooks/twilio_media_streams/tracker_stream",
      );
    });
  });

  describe("onopen", () => {
    it("sends retrieve action when sender query param is present", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" }, [
        "/?sender=existing-sender",
      ]);

      act(() => {
        lastWs?.onopen?.();
      });

      expect(lastWs?.send).toHaveBeenCalledWith(
        JSON.stringify({ action: "retrieve", sender_id: "existing-sender" }),
      );
    });

    it("does not send anything when no sender query param is present", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" });

      act(() => {
        lastWs?.onopen?.();
      });

      expect(lastWs?.send).not.toHaveBeenCalled();
    });
  });

  describe("onmessage", () => {
    it("updates the store with sessionId, events, stack and slots on a valid payload", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" });

      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "abc-123",
            events: [
              {
                event: "slot",
                name: "some_slot",
                value: "some_value",
                timestamp: 1,
              },
            ],
            slots: {},
            stack: [
              {
                frame_id: "f1",
                flow_id: "my_flow",
                step_id: "s1",
              },
            ],
          }),
        });
      });

      expect(inspectorStore.state.sessionId).toBe("abc-123");
      expect(inspectorStore.state.conversationList).toHaveLength(1);
      expect(inspectorStore.state.conversationList[0].id).toBe("abc-123");
      expect(inspectorStore.state.stack).toEqual([
        {
          frameId: "f1",
          flowId: "my_flow",
          stepId: "s1",
          collect: undefined,
          utter: undefined,
          ended: false,
        },
      ]);
      expect(inspectorStore.state.slots).toHaveLength(1);
      expect(inspectorStore.state.slots[0].name).toBe("some_slot");
      expect(inspectorStore.state.inputDisabled).toBe(true);
    });

    it("keeps the previous stack when the new stack is empty", () => {
      const previousStack = [
        {
          frameId: "prev",
          flowId: "prev_flow",
          stepId: "prev_step",
          ended: false,
        },
      ];

      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
          stack: previousStack,
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" });

      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "abc-123",
            events: [],
            slots: {},
            stack: [],
          }),
        });
      });

      expect(inspectorStore.state.stack).toEqual(previousStack);
    });

    it("ignores malformed JSON payloads", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      const initialSessionId = inspectorStore.state.sessionId;
      renderTrackerHook({ enabled: true, channel: "rest" });

      act(() => {
        lastWs?.onmessage?.({ data: "not-json{" });
      });

      expect(inspectorStore.state.sessionId).toBe(initialSessionId);
    });

    it("ignores payloads that fail schema validation", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      const initialSessionId = inspectorStore.state.sessionId;
      renderTrackerHook({ enabled: true, channel: "rest" });

      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({ totally: "invalid" }),
        });
      });

      expect(inspectorStore.state.sessionId).toBe(initialSessionId);
    });
  });

  describe("sender URL param sync", () => {
    it("locks the sender query param to the first sender_id seen when URL is empty", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      const { capturedSearch } = renderWithSearchParamsProbe(["/"]);

      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "first-sender",
            events: [],
            slots: {},
            stack: [],
          }),
        });
      });

      expect(capturedSearch.current).toBe("first-sender");
    });

    it("keeps the locked sender_id when a message arrives for a different sender_id", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      const { capturedSearch } = renderWithSearchParamsProbe([
        "/?sender=locked-sender",
      ]);

      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "other-sender",
            events: [],
            slots: {},
            stack: [],
          }),
        });
      });

      expect(capturedSearch.current).toBe("locked-sender");
    });
  });

  describe("conversation isolation", () => {
    it("does not update the store when a message arrives for a different sender_id", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
          sessionId: "locked-sender",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" }, [
        "/?sender=locked-sender",
      ]);

      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "other-sender",
            events: [
              {
                event: "slot",
                name: "some_slot",
                value: "some_value",
                timestamp: 1,
              },
            ],
            slots: {},
            stack: [
              {
                frame_id: "f1",
                flow_id: "my_flow",
                step_id: "s1",
              },
            ],
          }),
        });
      });

      expect(inspectorStore.state.sessionId).toBe("locked-sender");
      expect(inspectorStore.state.conversationList).toHaveLength(0);
      expect(inspectorStore.state.stack).toEqual([]);
      expect(inspectorStore.state.slots).toEqual([]);
    });

    it("locks to the first sender synchronously so rapid second message with another sender is ignored", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" });

      // Two messages in the same tick — the first locks the sender, the
      // second must not be able to override before React re-renders.
      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "first-sender",
            events: [],
            slots: {},
            stack: [],
          }),
        });
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "second-sender",
            events: [],
            slots: {},
            stack: [],
          }),
        });
      });

      expect(inspectorStore.state.sessionId).toBe("first-sender");
      expect(inspectorStore.state.conversationList).toHaveLength(1);
      expect(inspectorStore.state.conversationList[0].id).toBe(
        "first-sender",
      );
    });
  });

  describe("WebSocket lifecycle", () => {
    it("does not recreate the WebSocket when the sender URL param changes", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      renderTrackerHook({ enabled: true, channel: "rest" }, ["/"]);

      expect(wsCtor).toHaveBeenCalledTimes(1);

      // First message locks the URL to "first-sender". The WS must NOT be
      // torn down and rebuilt as a result of that URL change.
      act(() => {
        lastWs?.onmessage?.({
          data: JSON.stringify({
            sender_id: "first-sender",
            events: [],
            slots: {},
            stack: [],
          }),
        });
      });

      expect(wsCtor).toHaveBeenCalledTimes(1);
    });
  });

  describe("cleanup", () => {
    it("closes the WebSocket when the hook unmounts", () => {
      act(() => {
        inspectorStore.setState((prev) => ({
          ...prev,
          projectUrl: "http://localhost:5005",
        }));
      });

      const { unmount } = renderTrackerHook({
        enabled: true,
        channel: "rest",
      });

      const ws = lastWs;
      expect(ws?.close).not.toHaveBeenCalled();

      unmount();

      expect(ws?.close).toHaveBeenCalled();
    });
  });
});
