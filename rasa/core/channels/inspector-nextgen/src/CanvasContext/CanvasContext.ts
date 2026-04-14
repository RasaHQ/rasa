import { createContext } from 'react';
import type { Edge, Node, NodeChange, ReactFlowInstance } from 'reactflow';
import type { FlowNode } from '../types';

export interface ContextReturnType {
  flowName: string;
  nodes: Node<FlowNode>[];
  edges: Edge[];
  handleInit: (instance: ReactFlowInstance) => void;
  handleResize: () => void;
  focusOnNode: (node?: Node | null, animate?: boolean) => void;
  handleNodesChange: (changes: NodeChange[]) => void;
  handleZoomInClick: () => void;
  handleZoomOutClick: () => void;
  handleZoomTo100: () => void;
  handleFitToCanvasClick: () => void;
}

const defaultValues: ContextReturnType = {
  flowName: "",
  nodes: [],
  edges: [],
  handleInit: () => null,
  handleResize: () => null,
  focusOnNode: () => null,
  handleNodesChange: () => null,
  handleZoomInClick: () => null,
  handleZoomOutClick: () => null,
  handleZoomTo100: () => null,
  handleFitToCanvasClick: () => null,
};

export const Context = createContext<ContextReturnType>(defaultValues);
