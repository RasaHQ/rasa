import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => {
  const inspectorCanvas = page.getByTestId("inspector-canvas");
  const flowTimelineItems = inspectorCanvas.getByTestId("flow-timeline-item");

  return {
    flowTimeline: inspectorCanvas.getByTestId("flow-timeline"),
    flowTimelineItems,
    flowTimelineItemByName: (name: string) =>
      flowTimelineItems.filter({ hasText: name }),
    historyPlaceholder: inspectorCanvas.getByText(
      "Conversation history will be shown here.",
    ),
  };
};

export const actions = (_page: Page) => ({});

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    placeholderIsVisible: async () => {
      await expect(
        locators.historyPlaceholder,
        "Conversation history placeholder should be visible",
      ).toBeVisible();
    },
    timelineIsVisible: async () => {
      await expect(
        locators.flowTimeline,
        "Conversation history timeline should be visible",
      ).toBeVisible({ timeout: 10000 });
    },
    timelineIsHidden: async () => {
      await expect(
        locators.flowTimeline,
        "Conversation history timeline should be hidden",
      ).toBeHidden();
    },
    itemCountIs: async (count: number) => {
      await expect(
        locators.flowTimelineItems,
        `Conversation history timeline should have ${count} entries`,
      ).toHaveCount(count, { timeout: 10000 });
    },
    itemIsVisible: async (flowName: string) => {
      await expect(
        locators.flowTimelineItemByName(flowName),
        `Conversation history timeline should include "${flowName}"`,
      ).toBeVisible({ timeout: 10000 });
    },
    itemIsHidden: async (flowName: string) => {
      await expect(
        locators.flowTimelineItemByName(flowName),
        `Conversation history timeline should not include "${flowName}"`,
      ).toBeHidden({ timeout: 10000 });
    },
    itemHasStatus: async (flowName: string, status: string) => {
      await expect(
        locators.flowTimelineItemByName(flowName),
        `Conversation history timeline entry "${flowName}" should show "${status}"`,
      ).toContainText(status);
    },
  };
};
