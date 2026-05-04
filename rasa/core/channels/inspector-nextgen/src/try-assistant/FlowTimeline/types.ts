export type FlowInvocationStatus =
  | "active"
  | "interrupted"
  | "completed"
  | "cancelled";

export type TimelineEntryType = "flow" | "agent";

export type FlowTimelineEntry = {
  id: string;
  type: TimelineEntryType;
  flowId: string;
  flowName?: string;
  agentId?: string;
  status: FlowInvocationStatus;
  startTime: Date;
  exactStartTime: number;
  endTime?: Date;
};
