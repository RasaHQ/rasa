export * from "./api";
export * from "./conversation";
export * from "./flow";
export * from "./inspector";
import type { RasaProError } from "./conversation";

export enum PlaceholderImage {
  NoFlow,
  Mobile,
  Cubes,
  CubesError,
  CubesPause,
}

export type LogErrorHint = {
  tags?: { component: string; action: string };
  extra?: Record<string, string | null>;
};

export type LogErrorFn = (error: unknown, hint?: LogErrorHint) => void;

export type TrackFn = (
  event: string,
  properties?: Record<string, unknown>,
) => void | Promise<void>;

export type ShowToastOptions = {
  title: string;
  description: string;
  type: "error" | "info" | "success" | "warning";
  duration?: number;
  closable?: boolean;
};

export type ShowToastFn = (options: ShowToastOptions) => void;

export type VoiceErrorHandler = ((err: RasaProError) => void) | null;

export type OnboardingTooltipTarget = "messageInput" | "inspectToggle";

export interface OnboardingTooltipConfig {
  target: OnboardingTooltipTarget;
  title: string;
  description: string;
  actionLabel: string;
  counter?: string;
  onAction: () => void;
  onDismiss: () => void;
}
