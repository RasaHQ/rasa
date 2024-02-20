export interface Slot {
  name: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: any;
}

export interface Event {
  event: "user" | "bot";
  text?: string;
  metadata?: { utter_action?: string };
}

export interface SelectedStack {
  stack: Stack;
  isUserSelected: boolean;
}

export interface Stack {
  frame_id: string;
  flow_id: string;
  step_id: string;
  collect?: string;
  utter?: string;
}

export interface Flow {
  id: string;
  description: string;
  name: string;
  steps: Step[];
}

interface NextStepThen {
  action: string;
  id: string;
  next: string;
  set_slots: unknown[];
}

interface NextStepIf {
  if: string;
  then: NextStepThen[];
}

interface NextStepElse {
  if: string;
  then: NextStepThen[];
  else: string;
}

export type NextStep = NextStepIf | NextStepElse;

interface Step {
  ask_before_filling: boolean;
  collect: string;
  action: string;
  link: string;
  description: string;
  id: string;
  next: string | NextStep[];
  reset_after_flow_ends: boolean;
  utter: string;
  set_slots?: unknown;
}
