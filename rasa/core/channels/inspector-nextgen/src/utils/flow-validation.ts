// Copied from Studio's validation-modern-assistant.ts with deletions
import { z } from "zod";

import type { NodeConditionSubType, NodeType } from "../types";

export enum FlowStepType {
  Collect,
  Action,
  Link,
  SetSlots,
  Call,
  Noop,
}

export interface FlowStepThen {
  if: string;
  then: FlowStep[] | string;
}

export interface FlowStepElse {
  else: FlowStep[] | string;
}

export interface FlowCollectStepRejection {
  if: string;
  utter: string;
}

export interface FlowStep {
  stepType?: FlowStepType;
  collect?: string;
  action?: string;
  set_slots?: Record<string, string | boolean | number | null>[];
  link?: string;
  call?: string;
  id?: string;
  utter?: string;
  noop?: boolean;
  askBeforeFilling?: boolean;
  forceSlotFilling?: boolean;
  silenceTimeout?: number;
  description?: string;
  resetAfterFlowEnds?: boolean;
  rejections?: FlowCollectStepRejection[];
  next?: FlowStep[] | string | FlowStepCondition[];
  mcpServer?: string;
}

export function isFlowStep(
  next: FlowStep | FlowStepCondition,
): next is FlowStep {
  return (next as FlowStep).stepType !== undefined;
}

export function isFlowStepThen(
  next: FlowStepCondition | FlowStep,
): next is FlowStepThen {
  return (
    (next as FlowStepThen).if !== undefined &&
    (next as FlowStepThen).then !== undefined
  );
}

export function isFlowStepElse(
  next: FlowStepCondition | FlowStep,
): next is FlowStepElse {
  return (next as FlowStepElse).else !== undefined;
}

export type FlowStepCondition = FlowStepThen | FlowStepElse;

type FlowStepWithoutStepType = Omit<FlowStep, "stepType">;
export type PreNodeFlowStep = FlowStepWithoutStepType & {
  nodeType?: NodeType;
  condition?: string;
  conditionType?: NodeConditionSubType;
  conditionOrder?: number;
};

interface NluTrigger {
  intent: {
    name: string;
    confidence_threshold: number;
  };
}

interface FlowNameTranslation {
  name: string;
}

interface Flow {
  name?: string;
  translation?: Record<string, FlowNameTranslation>;
  description: string;
  persisted_slots?: string[];
  steps: FlowStep[];
  nlu_trigger?: NluTrigger[];
  if?: string | boolean;
  always_include_in_prompt?: boolean;
}

export interface Flows {
  flows: Record<string, Flow>;
}

const RejectionSchema = z.object({
  if: z.string(),
  utter: z.string(),
});

const baseStepSchema = z
  .object({
    noop: z.boolean(),
    collect: z.string(),
    action: z.string(),
    link: z.string(),
    set_slots: z.array(
      z.record(
        z.string(),
        z.union([z.string(), z.boolean(), z.number(), z.null()]),
      ),
    ),
    call: z.string(),
    id: z.string(),
    utter: z.string(),
    rejections: z.array(RejectionSchema),
    ask_before_filling: z.boolean(),
    force_slot_filling: z.boolean(),
    silence_timeout: z.number(),
    reset_after_flow_ends: z.boolean(),
    description: z.string(),
    mcp_server: z.string(),
  })
  .partial();

type Step = z.infer<typeof baseStepSchema> & {
  next?:
  | Step[]
  | string
  | ({ if: string; then: Step[] | string } | { else: Step[] | string })[];
};

const stepSchema: z.ZodType<Step> = baseStepSchema
  .extend({
    next: z
      .lazy(() =>
        z.union([
          z.array(stepSchema),
          z.string(),
          z.array(
            z.union([
              z.object({
                if: z.string(),
                then: z.union([z.array(stepSchema), z.string()]),
              }),
              z.object({
                else: z.union([z.array(stepSchema), z.string()]),
              }),
            ]),
          ),
        ]),
      )
      .optional(),
  })
  .refine(
    (step) =>
      (step.noop &&
        !step.collect &&
        !step.action &&
        !step.link &&
        !step.set_slots &&
        !step.call) ||
      (step.collect &&
        !step.action &&
        !step.link &&
        !step.set_slots &&
        !step.call &&
        !step.noop) ||
      (step.action &&
        !step.collect &&
        !step.link &&
        !step.set_slots &&
        !step.call &&
        !step.noop) ||
      (step.set_slots &&
        !step.collect &&
        !step.action &&
        !step.link &&
        !step.call &&
        !step.noop) ||
      (step.link &&
        !step.collect &&
        !step.action &&
        !step.set_slots &&
        !step.call &&
        !step.noop) ||
      (step.call &&
        !step.collect &&
        !step.action &&
        !step.link &&
        !step.set_slots &&
        !step.noop),
  )
  .transform((step) => {
    const transformedStep = {
      ...step,
    };
    if (step.collect) {
      return {
        ...transformedStep,
        askBeforeFilling: step.ask_before_filling,
        forceSlotFilling: step.force_slot_filling,
        resetAfterFlowEnds: step.reset_after_flow_ends,
        silenceTimeout: step.silence_timeout,
      };
    }
    return transformedStep;
  });

const nluTriggerSchema = z.object({
  intent: z
    .string()
    .transform((name) => ({
      name,
      confidence_threshold: 0.0,
    }))
    .or(
      z.object({
        name: z.string(),
        confidence_threshold: z.number().optional().default(0.0),
      }),
    ),
});

const flowTranslationSchema = z.record(
  z.string(),
  z.object({
    name: z.string(),
  }),
);

const flowSchema = z.object({
  name: z.string().optional(),
  description: z.string(),
  translation: flowTranslationSchema.optional(),
  persisted_slots: z
    .array(z.string())
    .nullable()
    .optional()
    .transform((val) => (val === null ? undefined : val)),
  steps: z.array(stepSchema),
  nlu_trigger: z
    .array(nluTriggerSchema)
    .nullable()
    .optional()
    .transform((val) => (val === null ? undefined : val)),
  if: z.string().or(z.boolean()).optional(),
  always_include_in_prompt: z.boolean().optional(),
  file_path: z.string().optional(),
});

// This is new for Hello Rasa
export type BotData = {
  flows: Record<string, Flow>;
};

export const BotDataSchema = z.object({
  flows: z.record(z.string(), flowSchema),
});
