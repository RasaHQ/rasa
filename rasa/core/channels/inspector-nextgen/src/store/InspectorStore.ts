import type { RefObject } from "react";
import { Store } from "@tanstack/store";
import type {
  Conversation,
  ConversationEvent,
  ConversationEventAction,
  Flow,
  SlotState,
  Stack,
  UnionEventType,
  VoiceErrorHandler,
} from "../types";
import  { InspectorView } from "../types/inspector";

export type InspectorStoreState = {
  // Connection / tracker data
  sessionId: string;
  conversationList: Conversation[];
  stack: Stack[];
  inputDisabled: boolean;
  replayingConversation: boolean;
  waitingForUserInput: boolean;
  slots: SlotState[];
  slotRelatedEvents: ConversationEvent[];

  // UI state
  inspectMode: boolean;
  inspectorView: InspectorView;
  selectedElement: UnionEventType | undefined;

  // Flows data
  flows: Flow[];
  flowsLoading: boolean;
  flowsError: Error | null;

  // Configuration
  projectUrl: string;
  botDataEndpoint: string;
  conversationEventActions: ConversationEventAction[];
  voiceFeaturesEnabled: boolean;

  // Actions (populated by useBotConnection after socket setup)
  sendMessage: (message: string) => void;
  startNewConversation: () => void;
  replayConversation: (events: UnionEventType[]) => void;
  setUrl: (url: string) => void;
  startVoiceStreaming: () => Promise<void>;
  stopVoiceStreaming: () => Promise<void>;
  onVoiceErrorRef: RefObject<VoiceErrorHandler>;
};

export type InspectorStore = Store<InspectorStoreState>;

// eslint-disable-next-line @typescript-eslint/no-empty-function
const noop = () => {};
const asyncNoop = () => Promise.resolve();

export function createInspectorStore(
  initialState?: Partial<InspectorStoreState>,
): InspectorStore {
  return new Store<InspectorStoreState>({
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

    ...initialState,
  });
}

export let inspectorStore: InspectorStore = createInspectorStore();

export function initInspectorStore(
  initialState?: Partial<InspectorStoreState>,
): InspectorStore {
  inspectorStore = createInspectorStore(initialState);
  return inspectorStore;
}
