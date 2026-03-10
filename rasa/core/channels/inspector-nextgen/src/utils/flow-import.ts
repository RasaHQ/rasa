// Copied from Studio's backend/server/services/cms/importing/flows.ts
import { partition, startsWith } from "lodash";
import {
  type FlowNode,
  type FlowEdge,
  FlowEdgeType,
  FlowNodeType,
  NodeType,
  NodeConditionSubType,
} from "../types";
import {
  isFlowStepThen,
  FlowStepType,
  type FlowStep,
  type FlowStepCondition,
  type PreNodeFlowStep,
} from "./flow-validation";

// Copy from Studio
function stepTypeToNodeType(stepType: FlowStepType): NodeType | undefined {
  switch (stepType) {
    case FlowStepType.Collect:
      return NodeType.COLLECT_INFORMATION;
    case FlowStepType.Link:
      return NodeType.LINK;
    case FlowStepType.Action: // Include NodeType.MESSAGE
      return NodeType.CUSTOM_ACTION;
    case FlowStepType.SetSlots:
      return NodeType.SET_SLOTS;
    case FlowStepType.Call:
      return NodeType.CALL;
    case FlowStepType.Noop:
      return;
    default:
      return;
  }
}

// Modified from Studio
function stepToPreNodeStep(step: FlowStep): PreNodeFlowStep {
  let stepType = step.stepType;
  if (stepType === undefined) {
    const key = Object.keys(step).find((k) =>
      ["collect", "action", "set_slots", "link", "call", "noop"].includes(k),
    );

    switch (key) {
      case "collect":
        stepType = FlowStepType.Collect;
        break;
      case "action":
        stepType = FlowStepType.Action;
        break;
      case "set_slots":
        stepType = FlowStepType.SetSlots;
        break;
      case "link":
        stepType = FlowStepType.Link;
        break;
      case "call":
        stepType = FlowStepType.Call;
        break;
      case "noop":
        stepType = FlowStepType.Noop;
        break;
      default:
        throw new Error(
          `Cannot determine step type for step: ${JSON.stringify(step, null, 2)}`,
        );
    }
  }

  return {
    ...step,
    nodeType: stepTypeToNodeType(stepType),
  };
}

// Copy from Studio
function stepNextIsLogic(
  stepNext: string | FlowStep[] | FlowStepCondition[],
): stepNext is FlowStepCondition[] {
  return typeof stepNext === "object" && "if" in stepNext[0];
}

// Copy from Studio
export function generateLogicStepIdForStep(step: {
  noop?: boolean;
  id?: string;
}) {
  return step.noop && step.id ? step.id : crypto.randomUUID();
}

// Copy from Studio
export function buildFlowStructureFromSteps({
  steps,
  processedSteps,
  stepSources,
  conditionStepSources,
}: {
  steps: PreNodeFlowStep[];
  processedSteps: (PreNodeFlowStep & { id: string })[];
  stepSources: Record<string, (PreNodeFlowStep | null)[]>;
  conditionStepSources: Record<string, string>;
}): {
  steps: (PreNodeFlowStep & { id: string })[];
  edges: Record<string, (PreNodeFlowStep | null)[]>;
  conditionStepSources: Record<string, string>;
} {
  const [step] = steps;
  if (!step) {
    return { steps: processedSteps, edges: stepSources, conditionStepSources };
  }
  let rest = steps.slice(1);
  const stepWithId = {
    ...step,
    id: step.id || crypto.randomUUID(),
  };
  const localStepSources = { ...stepSources };
  const localConditionStepSources = { ...conditionStepSources };
  if (!processedSteps.length) {
    localStepSources[stepWithId.id] = [null];
  }
  if (!step.id || !localStepSources[step.id]) {
    const sourceStep = processedSteps[processedSteps.length - 1];
    localStepSources[stepWithId.id] = [sourceStep];
  }
  if (step.next && step.next !== "END") {
    if (typeof step.next === "string") {
      localStepSources[step.next] = [
        ...(localStepSources[step.next] || []),
        stepWithId,
      ];
      // process steps in the order of the flow, not in the order they are defined
      const [nextStep, restWithoutNext] = partition(rest, { id: step.next });
      if (nextStep?.[0]?.next) {
        rest = [...nextStep, ...restWithoutNext];
      }
    } else if (stepNextIsLogic(step.next)) {
      const ifId = generateLogicStepIdForStep(step);
      const conditionsArray = step.next;
      const allConditionSteps: (PreNodeFlowStep & { id: string })[] =
        conditionsArray.map((condition, index) => ({
          ...condition,
          nodeType: NodeType.CONDITION,
          condition: isFlowStepThen(condition) ? condition.if : undefined,
          conditionType: isFlowStepThen(condition)
            ? NodeConditionSubType.IF
            : NodeConditionSubType.ELSE,
          conditionOrder: index + 1,
          next: isFlowStepThen(condition) ? condition.then : condition.else,
          id: crypto.randomUUID(),
        }));
      const mainLogicStep: PreNodeFlowStep & { id: string } = {
        id: ifId,
        nodeType: NodeType.LOGIC,
      };
      const allLogicSteps = [mainLogicStep, ...allConditionSteps];
      for (const conditionStep of allConditionSteps) {
        localStepSources[conditionStep.id] = [mainLogicStep];
        localConditionStepSources[conditionStep.id] = mainLogicStep.id;
      }
      rest = [...allLogicSteps, ...rest];
    } else {
      rest = [...step.next.map(stepToPreNodeStep), ...rest];
    }
  }
  return buildFlowStructureFromSteps({
    steps: rest,
    processedSteps: [...processedSteps, stepWithId],
    stepSources: localStepSources,
    conditionStepSources: localConditionStepSources,
  });
}

// Modified from Studio
function createFlowNodeFromStep(
  step: PreNodeFlowStep & { id: string },
  logicNodeIdsPerConditionStep: Record<string, string>,
): FlowNode | undefined {
  const node = {
    id: step.id,
    metadata: { x: 0, y: 0 },
  };
  switch (step.nodeType) {
    case NodeType.COLLECT_INFORMATION:
      return {
        ...node,
        type: FlowNodeType.CollectInformation,
        label: step.collect ?? "Collect Information",
      };
    case NodeType.LINK:
      return {
        ...node,
        type: FlowNodeType.Link,
        label: step.link ?? "Link",
      };
    case NodeType.CUSTOM_ACTION:
      if (startsWith(step.action, "utter_")) {
        return {
          ...node,
          type: FlowNodeType.Message,
          label: step.action ?? "Message",
        };
      } else {
        return {
          ...node,
          type: FlowNodeType.CustomAction,
          label: step.action || "Custom Action",
        };
      }
    case NodeType.SET_SLOTS:
      return {
        ...node,
        type: FlowNodeType.SetSlots,
        label: "Set Slots",
      };
    case NodeType.CALL:
      return {
        ...node,
        type: FlowNodeType.Call,
        label: "Call a flow and return",
      };
    case NodeType.LOGIC:
      return {
        ...node,
        type: FlowNodeType.Logic,
        label: "If...",
      };
    case NodeType.CONDITION: {
      const isThen = step.conditionType === NodeConditionSubType.IF;
      const logicId = logicNodeIdsPerConditionStep[step.id];
      return {
        ...node,
        type: FlowNodeType.Condition,
        logicId,
        order: step.conditionOrder ?? 1,
        subType: isThen ? NodeConditionSubType.IF : NodeConditionSubType.ELSE,
        label: isThen
          ? `If ${step.condition || "Condition"}`
          : `Else ${step.condition || "Condition"}`,
      };
    }
    default:
      console.warn(`Unknown step type: ${JSON.stringify(step)}`);
      return undefined;
  }
}

// Modified from Studio
export function createFlowNodesFromFlowSteps(
  steps: FlowStep[],
  flowName: string,
): {
  nodes: FlowNode[];
  edges: FlowEdge[];
} {
  const {
    steps: processedSteps,
    edges,
    conditionStepSources,
  } = buildFlowStructureFromSteps({
    steps: steps.map(stepToPreNodeStep),
    processedSteps: [],
    stepSources: {},
    conditionStepSources: {},
  });
  const logicNodeIdsPerConditionStep: Record<string, string> = {};
  const nodesMapByStepId = new Map<string, FlowNode>();
  for (const step of processedSteps) {
    if (step.noop) {
      continue;
    }

    const node = createFlowNodeFromStep(step, logicNodeIdsPerConditionStep);
    if (node && node.type === FlowNodeType.Logic) {
      const conditionStepIds = Object.keys(conditionStepSources).filter(
        (conditionStepId) => conditionStepSources[conditionStepId] === step.id,
      );
      for (const conditionStepId of conditionStepIds) {
        logicNodeIdsPerConditionStep[conditionStepId] = node.id;
      }
    }
    if (!step.id) {
      throw new Error(`Step should always have an ID, flow: ${flowName}`);
    }
    if (node) {
      nodesMapByStepId.set(step.id, node);
    }
  }
  const edgesToCreate: FlowEdge[] = [];
  for (const stepId of Object.keys(edges)) {
    const step = processedSteps.find((s) => s.id === stepId && !s.noop);
    if (!step) {
      continue;
    }
    const targetNode = nodesMapByStepId.get(stepId);
    if (!targetNode) {
      throw new Error(`Target node for ${stepId} not found, flow: ${flowName}`);
    }
    const sourceSteps = edges[stepId];
    const sourceNodes = sourceSteps.map((sourceStep) => {
      if (sourceStep && !sourceStep.id) {
        throw new Error(
          `Source step should always have an ID, flow: ${flowName}`,
        );
      }
      return sourceStep ? nodesMapByStepId.get(sourceStep.id!) : null;
    });
    for (const [index, sourceNode] of sourceNodes.entries()) {
      if (sourceNode) {
        edgesToCreate.push({
          sourceId: sourceNode.id,
          targetId: targetNode.id,
          type: index === 0 ? FlowEdgeType.Auto : FlowEdgeType.Custom,
        });
      }
    }
  }

  // Create missing start node and edge
  const startNode: FlowNode = {
    id: "START",
    type: FlowNodeType.Start,
    label: "Start",
    metadata: {
      x: 0,
      y: 0,
    },
  };

  const allStepsAreNoop = processedSteps.every((step) => step.noop);
  if (allStepsAreNoop) {
    return {
      nodes: [startNode],
      edges: [],
    };
  }

  // Find the first non-noop step to connect the start edge to
  // Should never happen, but just in case
  const firstNonNoopStep = processedSteps.find((step) => !step.noop);
  if (!firstNonNoopStep) {
    throw new Error(
      `No valid target step found for start edge, flow: ${flowName}`,
    );
  }

  const startEdge: FlowEdge = {
    sourceId: startNode.id,
    targetId: firstNonNoopStep.id,
    type: FlowEdgeType.Auto,
  };

  return {
    nodes: [startNode, ...Array.from(nodesMapByStepId.values())],
    edges: [startEdge, ...edgesToCreate],
  };
}

