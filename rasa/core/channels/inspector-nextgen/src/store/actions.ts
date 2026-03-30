import type { UnionEventType } from "../types";
import { inspectorStore } from "./InspectorStore";
import type { InspectorView } from "../types/inspector"

export function toggleSelectedElement(element: UnionEventType): void {
  inspectorStore.setState((prev) => ({
    ...prev,
    selectedElement:
      element.id === prev.selectedElement?.id ? undefined : element,
  }));
}

export function clearSelectedElement(): void {
  inspectorStore.setState((prev) => ({
    ...prev,
    selectedElement: undefined,
  }));
}

export function setInspectMode(enabled: boolean): void {
  inspectorStore.setState((prev) => ({ ...prev, inspectMode: enabled }));
}

export function setInspectorView(view: InspectorView): void {
  inspectorStore.setState((prev) => ({ ...prev, inspectorView: view }));
}
