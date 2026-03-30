export type FlowInvocationStatus =
  | "active"
  | "interrupted"
  | "completed"
  | "cancelled";

export type FlowTimelineEntry = {
  id: string;
  flowId: string;
  flowName?: string;
  status: FlowInvocationStatus;
  startTime: Date;
  endTime?: Date;
};
