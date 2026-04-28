import { describe, expect, it } from "vitest";
import { CallType, FlowNodeType } from "../types";
import { FlowStepType, type FlowStep } from "./flow-validation";
import { createFlowNodesFromFlowSteps } from "./flow-import";

function callStep(
  callTarget: string,
  options: { id?: string; mcpServer?: string } = {},
): FlowStep {
  return {
    stepType: FlowStepType.Call,
    call: callTarget,
    id: options.id ?? "step-1",
    ...(options.mcpServer ? { mcpServer: options.mcpServer } : {}),
  };
}

function findCallNode(nodes: ReturnType<typeof createFlowNodesFromFlowSteps>["nodes"]) {
  return nodes.find((n) => n.type === FlowNodeType.Call);
}

describe("createFlowNodesFromFlowSteps – Call node", () => {
  describe("label", () => {
    it("uses 'Call <target>' as the label", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("my_agent")],
        "test_flow",
      );
      expect(findCallNode(nodes)?.label).toBe("Call my_agent");
    });
  });

  describe("callType", () => {
    it("defaults to Agent when flowNamesList is not provided", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("some_agent")],
        "test_flow",
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Agent });
    });

    it("defaults to Agent when flowNamesList is an empty array", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("some_agent")],
        "test_flow",
        [],
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Agent });
    });

    it("resolves to Agent when the call target is not in flowNamesList", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("some_agent")],
        "test_flow",
        ["payment_flow", "booking_flow"],
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Agent });
    });

    it("resolves to Flow when the call target matches an entry in flowNamesList", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("payment_flow")],
        "test_flow",
        ["payment_flow", "booking_flow"],
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Flow });
    });

    it("resolves to Tool when mcpServer is set and call target is not a flow", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("search_web", { mcpServer: "my_mcp_server" })],
        "test_flow",
        ["payment_flow"],
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Tool });
    });

    it("resolves to Tool when mcpServer is set and no flowNamesList", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("search_web", { mcpServer: "my_mcp_server" })],
        "test_flow",
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Tool });
    });

    it("Flow takes precedence over mcpServer when the call target matches a flow name", () => {
      const { nodes } = createFlowNodesFromFlowSteps(
        [callStep("payment_flow", { mcpServer: "my_mcp_server" })],
        "test_flow",
        ["payment_flow"],
      );
      expect(findCallNode(nodes)).toMatchObject({ callType: CallType.Flow });
    });
  });
});
