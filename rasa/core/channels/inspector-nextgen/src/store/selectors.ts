import type { InspectorStoreState } from "./InspectorStore";
import type { Stack } from "../types";

const ALLOWED_PATTERNS = ["pattern_session_start", "pattern_completed"];

function isUserVisibleFrame(frame: Stack) {
  return (
    !frame.flowId?.startsWith("pattern_") ||
    ALLOWED_PATTERNS.includes(frame.flowId)
  );
}

export const selectLatestStack = (state: InspectorStoreState) => {
  const { stack } = state;
  for (let i = stack.length - 1; i >= 0; i--) {
    if (isUserVisibleFrame(stack[i])) return stack[i];
  }
  return undefined;
};

export const selectStackToShow = (state: InspectorStoreState) => {
  const { selectedElement } = state;
  if (
    selectedElement?.metadata?.flow_id &&
    selectedElement?.metadata?.step_id
  ) {
    return {
      frameId: selectedElement.id,
      flowId: selectedElement.metadata.flow_id,
      stepId: selectedElement.metadata.step_id,
      ended: false,
    };
  }
  return selectLatestStack(state);
};

export const selectConversationForSelectedElement = (
  state: InspectorStoreState,
) => {
  const { selectedElement, conversationList } = state;
  if (!selectedElement?.id) return undefined;
  return conversationList.find((conversation) =>
    conversation.events.some((event) => event?.id === selectedElement.id),
  );
};

export const selectWaitingForResponse = (state: InspectorStoreState) =>
  !state.waitingForUserInput && !state.inputDisabled;
