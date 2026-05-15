import type { FlexProps } from "@chakra-ui/react";
import type { IconDefinition } from "@fortawesome/fontawesome-svg-core";
import z from "zod";

export type InspectorEventType = ConversationEvent | StackEvent | Utterance;

export type StackEvent = {
  __typename: "StackEvent";
  id: string;
  metadata: EventMetadata;
  timestamp: string;
  update?: string;
};

export enum ConversationEventType {
  Action = "ACTION",
  ActiveLoop = "ACTIVE_LOOP",
  AgentCancelled = "AGENT_CANCELLED",
  AgentCompleted = "AGENT_COMPLETED",
  AgentInterrupted = "AGENT_INTERRUPTED",
  McpToolExecuted = "MCP_TOOL_EXECUTED",
  AgentResumed = "AGENT_RESUMED",
  AgentStarted = "AGENT_STARTED",
  FlowCancelled = "FLOW_CANCELLED",
  FlowCompleted = "FLOW_COMPLETED",
  FlowInterrupted = "FLOW_INTERRUPTED",
  FlowResumed = "FLOW_RESUMED",
  FlowStarted = "FLOW_STARTED",
  Form = "FORM",
  ResetSlots = "RESET_SLOTS",
  Restart = "RESTART",
  SessionStarted = "SESSION_STARTED",
  Slot = "SLOT",
}

export type Conversation = {
  events: UnionEventType[];
  id: string;
  reviewed: boolean;
  startDate: string;
  endDate?: string;
  totalNumberOfUserMessages: number;
};

type UtteranceEntity = {
  confidence: number;
  endPosition: number;
  id: string;
  name: string;
  startPosition: number;
  value: string;
};

export enum UtteranceType {
  Bot = "BOT",
  User = "USER",
}

type Token = {
  end: number;
  start: number;
  text: string;
};
export type QuickReply = {
  imageUrl?: string;
  payload: string;
  title: string;
};

export type Button = {
  imageUrl?: string;
  payload: string;
  title: string;
};

export type ResponseData = {
  attachment?: string;
  buttons: Button[];
  custom?: Record<string, unknown>;
  elements?: string;
  image?: string;
  quickReplies: QuickReply[];
};

export type UtteranceIntent = {
  confidence: number;
  id: string;
  name: string;
};
export type ConversationEvent = {
  __typename: "ConversationEvent";
  actionText: string;
  conversationEventType: ConversationEventType;
  agentId?: string;
  flowId: string;
  id: string;
  metadata: EventMetadata;
  name: string;
  slotValue: string | null;
  stepId: string;
  timestamp: string;
  originalTimestamp: number;
};

export type Utterance = {
  __typename: "Utterance";
  commands?: unknown;
  id: string;
  metadata: EventMetadata;
  rephrase: boolean;
  rephrasePrompt: string | null;
  responseData?: ResponseData;
  text: string;
  timestamp: string;
  originalTimestamp: number;
  tokens: Token[];
  type: UtteranceType;
  intents?: UtteranceIntent[];
  entities?: UtteranceEntity[];
};

export type UnionEventType = ConversationEvent | Utterance;

export type RasaExecutionTime = {
  command_processor: number;
  prediction_loop: number;
};

export type VoiceLatency = {
  asr_latency_ms: number;
  rasa_processing_latency_ms: number;
  tts_complete_latency_ms: number;
  tts_first_byte_latency_ms: number;
};

export type EventMetadata = {
  active_flow?: string;
  utter_action?: string;
  flow_id?: string;
  execution_success?: boolean;
  execution_error_message?: string;
  execution_times?: RasaExecutionTime;
  voiceLatency?: VoiceLatency;
  step_id?: string;
  rawEvent?: RawEvent;
  reset?: boolean;
  was_modified_by_studio?: boolean;
  domain_ground_truth?: string;
  metadata?: {
    rephrase?: boolean;
    rephrasePrompt?: string;
  };
  tool_name?: string;
  tool_arguments?: Record<string, unknown>;
  tool_result?: unknown;
  tool_is_error?: boolean;
  tool_error_message?: string;
  parseData: unknown;
  description?: string;
  mcp_tools?: string[];
  excluded_mcp_tools?: string[];
  exit_conditions?: string[];
};

export type RawEvent = {
  name: string;
  metadata: EventMetadata;
  timestamp: number;
  tool_name?: string;
  agent_id?: string;
  arguments?: Record<string, unknown>;
  result?: unknown;
  is_error?: boolean;
  error_message?: string;
  event:
  | "user"
  | "stack"
  | "action"
  | "bot"
  | "slot"
  | "agent"
  | "agent_started"
  | "agent_completed"
  | "agent_interrupted"
  | "agent_cancelled"
  | "agent_resumed"
  | "mcp_tool_executed";
  conversation_id: string;
  text: string;
  data: BackendResponseData;
  update: string;
  flow_id: string;
  step_id: string;
  value: string;
  parse_data: {
    intent_ranking: MessageIntent[];
    commands?: string;
  };
};

export type ModelServiceError = {
  message: string;
  error: string;
  exception?: string;
};

export type RawStack = {
  frame_id: string;
  flow_id: string;
  step_id: string;
  collect?: string;
  utter?: string;
};

export type Stack = {
  frameId: string;
  flowId: string;
  stepId: string;
  collect?: string;
  utter?: string;
  ended: boolean;
};

export type SlotState = {
  name: string;
  value: unknown;
  event?: ConversationEvent;
};

export type MessageIntent = {
  name: string;
  confidence: number;
};

export type RasaProError = {
  message: string;
  error: string;
  exception?: string;
};

export type BackendResponseData = {
  attachment: {
    type: string;
    payload: {
      src: string;
      url: string;
      elements: string;
    };
  };
  quick_replies: (QuickReply & { image_url?: string })[];
  buttons: (Button & { image_url?: string })[];
  elements: string;
  image: string;
  text: string;
};

export type TrackerResponseData = {
  sender_id: string;
  events: RawEvent[];
  slots: Record<string, SlotState>;
  stack: RawStack[];
};

export type MessageProps = FlexProps & {
  utterance: Omit<Utterance, "entities">;
  previousUtterance?: Omit<Utterance, "entities">;
  isHighlighted?: boolean;
  isInteractive?: boolean;
  onQuickReply?: (payload: string) => void;
  selectable?: boolean;
  isSelected?: boolean;
  onMessageSelect?: (selection: Utterance) => void;
  inspectorMode?: boolean;
};

export type ConversationEventAction = {
  icon: IconDefinition;
  label: string;
  action: (event: UnionEventType) => void
};

export type MessagePropsWithTopIntentName = MessageProps & {
  topIntentName?: string;
  conversationEventActions?: ConversationEventAction[];
};

export const TrackerResponseSchema = z.object({
  sender_id: z.string(),
  // TODO: replace any
  events: z.any(),
  slots: z.object(),
  stack: z.array(z.any()),
});
