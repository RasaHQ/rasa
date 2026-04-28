export enum FlowNodeType {
  Call = "CALL",
  CollectInformation = "COLLECT_INFORMATION",
  Condition = "CONDITION",
  CustomAction = "CUSTOM_ACTION",
  Link = "LINK",
  Logic = "LOGIC",
  Message = "MESSAGE",
  SetSlots = "SET_SLOTS",
  Start = "START",
}

export enum NodeConditionSubType {
  IF = "IF",
  ELSE = "ELSE",
}

type FlowNodeBase = {
  id: string;
  label: string;
  type: FlowNodeType;
  metadata: {
    x: number;
    y: number;
  };
};

type FlowNodeCollectInformation = FlowNodeBase & {
  type: FlowNodeType.CollectInformation;
};

type FlowNodeMessage = FlowNodeBase & {
  type: FlowNodeType.Message;
};

type FlowNodeCondition = FlowNodeBase & {
  type: FlowNodeType.Condition;
  logicId: string;
  order: number;
  subType: NodeConditionSubType;
};

type FlowNodeCustomAction = FlowNodeBase & {
  type: FlowNodeType.CustomAction;
};

type FlowNodeLink = FlowNodeBase & {
  type: FlowNodeType.Link;
};

export type FlowNodeStart = FlowNodeBase & {
  label: string;
  type: FlowNodeType.Start;
};

type FlowNodeLogic = FlowNodeBase & {
  type: FlowNodeType.Logic;
};

type FlowNodeSetSlots = FlowNodeBase & {
  type: FlowNodeType.SetSlots;
};

export enum CallType {
  Flow,
  Agent,
  Tool,
}

export type FlowNodeCall = FlowNodeBase & {
  type: FlowNodeType.Call;
  callType: CallType;
};

export type FlowNode =
  | FlowNodeCollectInformation
  | FlowNodeMessage
  | FlowNodeCondition
  | FlowNodeCustomAction
  | FlowNodeLink
  | FlowNodeStart
  | FlowNodeLogic
  | FlowNodeSetSlots
  | FlowNodeCall;

export enum FlowEdgeType {
  Auto = "AUTO",
  Custom = "CUSTOM",
}

export type FlowEdge = {
  sourceId: string;
  targetId: string;
  type: FlowEdgeType;
};

// Flow from Rasa Studio
export type Flow = {
  id: string;
  name: string;
  description?: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
};

export enum NodeType {
  COLLECT_INFORMATION = "COLLECT_INFORMATION",
  MESSAGE = "MESSAGE",
  CONDITION = "CONDITION",
  CUSTOM_ACTION = "CUSTOM_ACTION",
  LINK = "LINK",
  START = "START",
  LOGIC = "LOGIC",
  SET_SLOTS = "SET_SLOTS",
  CALL = "CALL",
}
