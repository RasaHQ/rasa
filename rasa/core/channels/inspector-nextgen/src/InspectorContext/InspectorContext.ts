import { createContext } from "react";
import type {
  LogErrorFn,
  OnboardingTooltipConfig,
  ShowToastFn,
  TrackFn,
} from "../types";
import { defaultShowToast } from "../Toaster";

export interface InspectorContextValue {
  logError: LogErrorFn;
  track: TrackFn;
  showToast: ShowToastFn;
  onboardingTooltips: OnboardingTooltipConfig[];
}

export const InspectorContext = createContext<InspectorContextValue>({
  logError: console.error,
  track: () => undefined,
  showToast: defaultShowToast,
  onboardingTooltips: [],
});
