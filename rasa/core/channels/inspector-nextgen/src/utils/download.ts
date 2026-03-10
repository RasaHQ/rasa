import {
  type Conversation,
  type ConversationEvent,
  ConversationEventType,
  type RawEvent,
  type UnionEventType,
  type Utterance,
  UtteranceType,
} from "../types";
import { isConversationEvent, isUtterance } from "./conversation";

const INTERNAL_RASA_SLOTS = ["flow_hashes", "dialogue_stack"];

function isInternalRasaSlot(slotName: string): boolean {
  return INTERNAL_RASA_SLOTS.includes(slotName);
}

function generateFlowStartedAssertion(event: ConversationEvent): string {
  return `- flow_started: "${event.flowId}"`;
}

function generateFlowCompletedAssertion(event: ConversationEvent): string {
  return `- flow_completed:\n      flow_id: "${event.flowId}"`;
}

function generateFlowCancelledAssertion(event: ConversationEvent): string {
  return `- flow_cancelled:\n      flow_id: "${event.flowId}"`;
}

function generateSlotWasSetAssertion(event: ConversationEvent): string {
  if (isInternalRasaSlot(event.name ?? "")) {
    return "";
  }
  return `- slot_was_set:\n    - name: "${event.name}"\n      value: ${JSON.stringify(event.slotValue)}`;
}

function generateActionExecutedAssertion(event: ConversationEvent): string {
  return `- action_executed: "${event.actionText}"`;
}

function generateBotUtteredAssertion(utterance: Utterance): string {
  const utterName = utterance.metadata?.utter_action;
  const lines: string[] = [
    `- bot_uttered:`,
    utterName
      ? `    utter_name: "${utterName}"`
      : `    text_matches: "${utterance.text}"`,
  ];

  if (utterance.responseData?.quickReplies.length) {
    lines.push(`    buttons:`);
    for (const button of utterance.responseData.quickReplies) {
      lines.push(`      - title: "${button.title}"`);
      lines.push(`        payload: "${button.payload}"`);
    }
  }

  return lines.join("\n");
}

function eventToAssertion(event: Utterance | ConversationEvent): string {
  if (isConversationEvent(event)) {
    switch (event.conversationEventType) {
      case ConversationEventType.FlowStarted:
        return generateFlowStartedAssertion(event);
      case ConversationEventType.FlowCompleted:
        return generateFlowCompletedAssertion(event);
      case ConversationEventType.FlowCancelled:
        return generateFlowCancelledAssertion(event);
      case ConversationEventType.Slot:
        return generateSlotWasSetAssertion(event);
      case ConversationEventType.Action:
        return generateActionExecutedAssertion(event);
      default:
        return "";
    }
  } else if (isUtterance(event) && event.type === UtteranceType.Bot) {
    return generateBotUtteredAssertion(event);
  }
  return "";
}

type Step = {
  user: string;
  assertions: string[];
};

export function generateTestCase(
  sessionId: string,
  conversation: Conversation,
): string {
  const steps: Step[] = [];
  let currentStep: Step | null = null;
  let hasStarted = false;

  for (const event of conversation.events) {
    if (isUtterance(event) && event.type === UtteranceType.User) {
      hasStarted = true;

      if (currentStep) {
        steps.push(currentStep);
      }

      currentStep = {
        user: event.text,
        assertions: [],
      };
    } else if (hasStarted) {
      if (!currentStep) {
        currentStep = {
          user: "N/A",
          assertions: [],
        };
      }
      const assertion = eventToAssertion(event);
      if (assertion) {
        currentStep.assertions.push(assertion);
      }
    }
  }

  if (currentStep) {
    steps.push(currentStep);
  }

  const lines: string[] = [];
  lines.push(`test_cases:`);
  lines.push(`  - test_case: ${sessionId}`);
  lines.push(`    steps:`);
  for (const step of steps) {
    lines.push(`      - user: ${JSON.stringify(step.user)}`);
    if (step.assertions.length > 0) {
      lines.push(`        assertions:`);
      for (const assertion of step.assertions) {
        for (const line of assertion.split("\n")) {
          lines.push(`          ${line}`);
        }
      }
    }
  }

  return lines.join("\n");
}

function triggerDownload(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function extractRawEvents(conversations: Conversation[]): RawEvent[] {
  return conversations.flatMap((conversation) =>
    conversation.events
      .map((event: UnionEventType) => event.metadata?.rawEvent)
      .filter((raw): raw is RawEvent => raw != null),
  );
}

export function downloadConversation(
  conversations: Conversation[],
  sessionId: string,
) {
  const rawEvents = extractRawEvents(conversations);
  const json = JSON.stringify(rawEvents, null, 2);
  triggerDownload(json, `conversation-${sessionId}.json`, "application/json");
}

export function downloadE2eTests(
  conversations: Conversation[],
  sessionId: string,
) {
  const lastConversation = conversations[conversations.length - 1];
  if (!lastConversation) return;

  const yaml = generateTestCase(sessionId, lastConversation);
  triggerDownload(yaml, `e2e-test-${sessionId}.yml`, "text/yaml");
}
