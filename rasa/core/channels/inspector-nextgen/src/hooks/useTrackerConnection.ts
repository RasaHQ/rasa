import { useEffect, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import {
  type Conversation,
  type RawStack,
  type Stack,
  TrackerResponseDataSchema,
  type UnionEventType,
} from "../types";
import { validateData } from "../api";
import {
  formatSlots,
  getSlotRelatedEvents,
  isUtterance,
  mapRawEventsToConversationEvents,
} from "../utils";
import { inspectorStore, useInspectorStore } from "../store";

export function useTrackerConnection({ enabled, channel }: { enabled: boolean; channel: string }) {
  const projectUrl = useInspectorStore((s) => s.projectUrl);
  const [searchParams, setSearchParams] = useSearchParams();
  const wsRef = useRef<WebSocket | undefined>(undefined);

  useEffect(() => {
    if (!enabled || !projectUrl) return;

    let senderIdFromUrl = searchParams.get("sender") ?? "";

    const wsUrl =
      projectUrl.replace(/^http/, "ws") +
      `/webhooks/${channel}/tracker_stream`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      if (senderIdFromUrl) {
        ws.send(JSON.stringify({ action: "retrieve", sender_id: senderIdFromUrl }));
      }
    };

    ws.onmessage = (messageEvent) => {
      let response;
      try {
        response = validateData(JSON.parse(messageEvent.data as string), TrackerResponseDataSchema);
      } catch {
        return;
      }

      const { sender_id: senderIdFromRasa = "", events: rawEvents, stack: rawStack } = response;

      // Prevent the hook from processing messages from a different sender.
      // Once we decide on senderIdFromUrl, we don't switch it on every message
      // with a different senderIdFromRasa.
      if (senderIdFromUrl.length > 0 && senderIdFromUrl !== senderIdFromRasa) {
        return;
      }

      // First message we see locks to this sender synchronously and persist to the URL.
      if (senderIdFromUrl.length === 0 && senderIdFromRasa.length > 0) {
        // this prevents two rapid messages before react re-renders to update the URL twice
        senderIdFromUrl = senderIdFromRasa;
        setSearchParams((prev) => {
          prev.set("sender", senderIdFromRasa);
          return prev;
        }, { replace: true });
      }

      const events = mapRawEventsToConversationEvents(rawEvents);

      const convertedStack: Stack[] = rawStack.map((item: RawStack) => ({
        frameId: item.frame_id,
        flowId: item.flow_id,
        stepId: item.step_id,
        collect: item.collect,
        utter: item.utter,
        ended: false,
      }));

      const conversation: Conversation = {
        id: senderIdFromRasa,
        events,
        startDate: new Date().toISOString(),
        reviewed: false,
        totalNumberOfUserMessages: events.filter((e: UnionEventType) =>
          isUtterance(e),
        ).length,
      };

      inspectorStore.setState((prev) => ({
        ...prev,
        sessionId: senderIdFromRasa,
        conversationList: [conversation],
        stack: convertedStack.length > 0 ? convertedStack : prev.stack,
        slots: formatSlots(getSlotRelatedEvents(events)),
        slotRelatedEvents: getSlotRelatedEvents(events),
        inputDisabled: true,
      }));
    };

    return () => {
      ws.close();
      wsRef.current = undefined;
    };
    // searchParams / setSearchParams are intentionally NOT in deps: we want the
    // WS to be created exactly once per [enabled, projectUrl, channel] tuple and
    // not re-created every time the sender query param changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, projectUrl, channel]);
}
