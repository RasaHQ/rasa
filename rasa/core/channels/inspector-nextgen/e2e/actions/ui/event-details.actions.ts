import { expect, type Page } from "@playwright/test";

export const getLocators = (page: Page) => ({
  inspectorCanvas: page.getByTestId("inspector-canvas"),
  closeButton: page.getByTestId("event-details-close"),
  panelHeading: (title: string) => page.getByRole("heading", { name: title }),
  slotOrFlowName: page.getByTestId("event-slot-or-flow-name"),
  slotValue: page.getByTestId("event-slot-value"),
  actionEventInfo: page.getByTestId("action-event-info"),
  accordionTitle: (title: string) =>
    page.getByTestId("inspector-canvas").getByText(title, { exact: true }),
});

export const actions = (page: Page) => {
  const locators = getLocators(page);

  return {
    close: async () => {
      await locators.closeButton.click();
    },
  };
};

export const assertions = (page: Page) => {
  const locators = getLocators(page);

  return {
    panelIsVisible: async (title: string) => {
      await expect(
        locators.panelHeading(title),
        `Event details panel with title "${title}" should be visible`,
      ).toBeVisible();
    },
    panelIsHidden: async () => {
      await expect(
        locators.closeButton,
        "Event details close button should not be visible",
      ).toBeHidden();
    },
    slotOrFlowNameIsVisible: async (name: string) => {
      await expect(
        locators.slotOrFlowName,
        `Slot or flow name "${name}" should be visible`,
      ).toContainText(name);
    },
    slotValueIsVisible: async (value?: string) => {
      if (value === undefined) {
        await expect(
          locators.slotValue,
          "Slot value should be visible",
        ).toBeVisible();
        return;
      }
      await expect(
        locators.slotValue,
        `Slot value "${value}" should be visible`,
      ).toContainText(value);
    },
    actionEventInfoIsVisible: async () => {
      await expect(
        locators.actionEventInfo,
        "Action event info should be visible",
      ).toBeVisible();
    },
    panelContainsText: async (text: string) => {
      await expect(
        locators.inspectorCanvas,
        `Event details panel should contain text "${text}"`,
      ).toContainText(text);
    },
    accordionIsVisible: async (title: string) => {
      await expect(
        locators.accordionTitle(title),
        `Accordion item "${title}" should be visible`,
      ).toBeVisible();
    },
  };
};
