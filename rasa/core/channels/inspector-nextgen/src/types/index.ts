export * from "./api";
export * from "./conversation";
export * from "./flow";

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
