import { useMemo, type ReactNode } from "react";
import { defaultShowToast } from "../Toaster";
import type {
  LogErrorFn,
  OnboardingTooltipConfig,
  ShowToastFn,
  TrackFn,
} from "../types";
import { InspectorContext } from "./InspectorContext";

interface Props {
  logError?: LogErrorFn;
  track?: TrackFn;
  showToast?: ShowToastFn;
  onboardingTooltips?: OnboardingTooltipConfig[];
  socketReconnectAttempts?: number;
  children: ReactNode;
}

export const InspectorContextProvider = ({
  logError,
  track,
  showToast,
  onboardingTooltips,
  socketReconnectAttempts,
  children,
}: Props) => {
  const value = useMemo(
    () => ({
      logError: logError ?? console.error,
      track: track ?? (() => undefined),
      showToast: showToast ?? defaultShowToast,
      onboardingTooltips: onboardingTooltips ?? [],
      socketReconnectAttempts,
    }),
    [logError, track, showToast, onboardingTooltips, socketReconnectAttempts],
  );

  return (
    <InspectorContext.Provider value={value}>
      {children}
    </InspectorContext.Provider>
  );
};
