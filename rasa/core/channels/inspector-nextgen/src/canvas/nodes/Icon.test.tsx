import { describe, expect, it } from "vitest";
import { renderWithProviders } from "../../tests/utils";
import { CallType, FlowNodeType, type FlowNode } from "../../types";
import { Icon } from "./Icon";

function makeNode(overrides: Partial<FlowNode> & { type: FlowNode["type"] }): FlowNode {
  return {
    id: "node-1",
    label: "Test Node",
    metadata: { x: 0, y: 0 },
    ...overrides,
  } as FlowNode;
}

function callNode(callType: CallType): FlowNode {
  return makeNode({ type: FlowNodeType.Call, callType });
}

function getSvgIcon(container: HTMLElement) {
  return container.querySelector("svg");
}

describe("Icon (canvas node)", () => {
  describe("renders without error for each node type", () => {
    it("CollectInformation", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.CollectInformation })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Message", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.Message })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("CustomAction", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.CustomAction })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("SetSlots", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.SetSlots })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Start", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.Start })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Link", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.Link })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Logic", () => {
      const { container } = renderWithProviders(
        <Icon node={makeNode({ type: FlowNodeType.Logic })} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Condition", () => {
      const { container } = renderWithProviders(
        <Icon
          node={makeNode({
            type: FlowNodeType.Condition,
            logicId: "logic-1",
            order: 1,
            subType: "IF" as never,
          })}
        />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Call (Flow)", () => {
      const { container } = renderWithProviders(
        <Icon node={callNode(CallType.Flow)} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Call (Agent)", () => {
      const { container } = renderWithProviders(
        <Icon node={callNode(CallType.Agent)} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });

    it("Call (Tool)", () => {
      const { container } = renderWithProviders(
        <Icon node={callNode(CallType.Tool)} />,
      );
      expect(getSvgIcon(container)).toBeInTheDocument();
    });
  });

  describe("Call node icons differ by callType", () => {
    it("Flow uses arrow-right-arrow-left icon", () => {
      const { container } = renderWithProviders(
        <Icon node={callNode(CallType.Flow)} />,
      );
      expect(container.querySelector('svg[data-icon="arrow-right-arrow-left"]')).toBeInTheDocument();
    });

    it("Agent uses robot icon", () => {
      const { container } = renderWithProviders(
        <Icon node={callNode(CallType.Agent)} />,
      );
      expect(container.querySelector('svg[data-icon="robot"]')).toBeInTheDocument();
    });

    it("Tool uses wrench icon", () => {
      const { container } = renderWithProviders(
        <Icon node={callNode(CallType.Tool)} />,
      );
      expect(container.querySelector('svg[data-icon="wrench"]')).toBeInTheDocument();
    });

    it("Flow and Agent render different icons", () => {
      const { container: flowContainer } = renderWithProviders(
        <Icon node={callNode(CallType.Flow)} />,
      );
      const { container: agentContainer } = renderWithProviders(
        <Icon node={callNode(CallType.Agent)} />,
      );
      const flowIcon = flowContainer.querySelector("svg")?.getAttribute("data-icon");
      const agentIcon = agentContainer.querySelector("svg")?.getAttribute("data-icon");
      expect(flowIcon).not.toBe(agentIcon);
    });
  });
});
