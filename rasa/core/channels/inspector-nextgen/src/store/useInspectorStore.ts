import { useStore } from "@tanstack/react-store";
import { inspectorStore, type InspectorStoreState } from "./InspectorStore";

export function useInspectorStore<T>(
  selector: (state: InspectorStoreState) => T,
): T {
  return useStore(inspectorStore, selector);
}
