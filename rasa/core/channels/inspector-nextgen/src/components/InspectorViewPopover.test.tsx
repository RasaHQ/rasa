import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../tests/utils";
import { InspectorViewPopover } from "./InspectorViewPopover";
import { InspectorView } from "../types/inspector";

const mockIsLargeScreen = vi.fn(() => false);
vi.mock("../hooks/useIsLargeScreen", () => ({
  useIsLargeScreen: () => mockIsLargeScreen(),
}));

describe("InspectorViewPopover", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsLargeScreen.mockReturnValue(false);
  });

  it("renders the trigger button", () => {
    renderWithProviders(<InspectorViewPopover />);
    expect(
      screen.getByLabelText("Show inspector view options"),
    ).toBeInTheDocument();
  });

  it("always shows Active flow, Flow history, and Memory items", () => {
    renderWithProviders(<InspectorViewPopover />);
    expect(screen.getByTestId("view-menu-active-flow")).toBeInTheDocument();
    expect(screen.getByTestId("view-menu-flow-history")).toBeInTheDocument();
    expect(screen.getByTestId("view-menu-memory")).toBeInTheDocument();
  });

  it("does not show All option on small screens", () => {
    mockIsLargeScreen.mockReturnValue(false);
    renderWithProviders(<InspectorViewPopover />);
    expect(screen.queryByTestId("view-menu-all")).not.toBeInTheDocument();
  });

  it("shows All option on large screens", () => {
    mockIsLargeScreen.mockReturnValue(true);
    renderWithProviders(<InspectorViewPopover />);
    expect(screen.getByTestId("view-menu-all")).toBeInTheDocument();
  });

  it("highlights the currently selected view", () => {
    renderWithProviders(<InspectorViewPopover />, {
      initialStoreState: { inspectorView: InspectorView.Memory },
    });
    const memoryItem = screen.getByTestId("view-menu-memory");
    expect(memoryItem).toBeInTheDocument();
  });

  it("highlights All when All is the selected view on large screen", () => {
    mockIsLargeScreen.mockReturnValue(true);
    renderWithProviders(<InspectorViewPopover />, {
      initialStoreState: { inspectorView: InspectorView.All },
    });
    expect(screen.getByTestId("view-menu-all")).toBeInTheDocument();
  });
});
