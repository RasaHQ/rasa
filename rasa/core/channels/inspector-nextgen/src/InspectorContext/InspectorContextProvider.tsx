import { useMemo, type ReactNode } from "react";
import type { LogErrorFn, OnboardingTooltipConfig, TrackFn } from "../types";
import { InspectorContext } from "./InspectorContext";

interface Props {
  logError?: LogErrorFn;
  track?: TrackFn;
  onboardingTooltips?: OnboardingTooltipConfig[];
  children: ReactNode;
}

export const InspectorContextProvider = ({
  logError,
  track,
  onboardingTooltips,
  children,
}: Props) => {
  const value = useMemo(
    () => ({
      logError: logError ?? console.error,
      track: track ?? (() => undefined),
      onboardingTooltips: onboardingTooltips ?? [],
    }),
    [logError, track, onboardingTooltips],
  );

  return (
    <InspectorContext.Provider value={value}>
      {children}
    </InspectorContext.Provider>
  );
};
