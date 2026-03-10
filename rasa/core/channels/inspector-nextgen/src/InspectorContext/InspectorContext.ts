import { createContext } from "react";
import type { LogErrorFn, OnboardingTooltipConfig, TrackFn } from "../types";

export interface InspectorContextValue {
  logError: LogErrorFn;
  track: TrackFn;
  onboardingTooltips: OnboardingTooltipConfig[];
}

export const InspectorContext = createContext<InspectorContextValue>({
  logError: console.error,
  track: () => undefined,
  onboardingTooltips: [],
});
