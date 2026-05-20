import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => {
  const inspectorCanvas = page.getByTestId("inspector-canvas");

  return {
    inspectorCanvas,
    canvas: page.getByTestId("canvas"),
    flowNodes: page.getByTestId("node"),
    flowNode: (nodeName: string) =>
      inspectorCanvas.getByTestId("node").filter({ hasText: nodeName }),
    flowName: (flowName: string) => inspectorCanvas.getByText(flowName),
    noActiveFlowPlaceholder: page.getByText("No flow is currently active"),
  };
};

export const actions = (_page: Page) => ({});

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    inspectorCanvasIsVisible: async () => {
      await expect(
        locators.inspectorCanvas,
        "Inspector canvas should be visible when Inspect is on",
      ).toBeVisible();
    },
    inspectorCanvasIsHidden: async () => {
      await expect(
        locators.inspectorCanvas,
        "Inspector canvas should be hidden when Inspect is off",
      ).toBeHidden();
    },
    flowPanelShowsNoActiveFlow: async () => {
      await expect(
        locators.noActiveFlowPlaceholder,
        "Flow panel should show the no-active-flow placeholder",
      ).toBeVisible();
    },
    canvasWithNodesIsVisible: async () => {
      await expect(locators.canvas, "Flow canvas should be visible").toBeVisible();
      await expect(
        locators.flowNodes.first(),
        "At least one flow node should be visible",
      ).toBeVisible({ timeout: 10000 });
    },
    flowNodeIsVisible: async (nodeName: string) => {
      await expect(
        locators.flowNode(nodeName),
        `Flow node "${nodeName}" should be visible`,
      ).toBeVisible({ timeout: 10000 });
    },
    activeFlowNameIsVisible: async (flowName: string) => {
      await expect(
        locators.flowName(flowName),
        `Active flow name "${flowName}" should be visible`,
      ).toBeVisible();
    },
    flowCanvasIsReplacedByDetails: async () => {
      await expect(
        locators.canvas,
        "Flow canvas should be hidden when the event details panel is shown",
      ).toBeHidden();
    },
    flowCanvasIsVisible: async () => {
      await expect(locators.canvas, "Flow canvas should be visible").toBeVisible();
    },
  };
};
