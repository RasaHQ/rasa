import { useMemo, useSyncExternalStore } from "react";
import { useInspectorStore } from "../store";
import { breakpoints } from "../theme/tokens/breakpoints";

export function useIsLargeScreen(): boolean {
  const isEmbedded = useInspectorStore((s) => s.isEmbedded);
  const query = isEmbedded
    ? `(min-width: ${breakpoints["3xl"]})`
    : `(min-width: ${breakpoints["2xl"]})`;

  const mql = useMemo(() => window.matchMedia(query), [query]);

  return useSyncExternalStore(
    (callback) => {
      mql.addEventListener("change", callback);
      return () => mql.removeEventListener("change", callback);
    },
    () => mql.matches,
  );
}
