import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../tests/utils";
import { InspectorViewHeader } from "./InspectorViewHeader";

vi.mock("./InspectorViewPopover", () => ({
  InspectorViewPopover: () => <div data-testid="view-popover" />,
}));

describe("InspectorViewHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders the title", () => {
    renderWithProviders(<InspectorViewHeader title="Flow history" />);
    expect(screen.getByText("Flow history")).toBeInTheDocument();
  });

  it("renders optional text when provided", () => {
    renderWithProviders(
      <InspectorViewHeader title="Active flow" text="my_flow" />,
    );
    expect(screen.getByText("Active flow")).toBeInTheDocument();
    expect(screen.getByText("my_flow")).toBeInTheDocument();
  });

  it("does not render optional text when omitted", () => {
    renderWithProviders(<InspectorViewHeader title="Memory" />);
    expect(screen.getByText("Memory")).toBeInTheDocument();
    expect(screen.queryByText("my_flow")).not.toBeInTheDocument();
  });

  it("shows the view popover by default", () => {
    renderWithProviders(<InspectorViewHeader title="Test" />);
    expect(screen.getByTestId("view-popover")).toBeInTheDocument();
  });

  it("hides the view popover when showViewSwitcher is false", () => {
    renderWithProviders(
      <InspectorViewHeader title="Test" showViewSwitcher={false} />,
    );
    expect(screen.queryByTestId("view-popover")).not.toBeInTheDocument();
  });

  it("shows the view popover when showViewSwitcher is true", () => {
    renderWithProviders(
      <InspectorViewHeader title="Test" showViewSwitcher />,
    );
    expect(screen.getByTestId("view-popover")).toBeInTheDocument();
  });
});
