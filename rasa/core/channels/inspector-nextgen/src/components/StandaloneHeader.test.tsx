import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../tests/utils";
import { StandaloneHeader } from "./StandaloneHeader";
import { inspectorStore } from "../store";

describe("StandaloneHeader", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders Chat and Inspect options", () => {
    renderWithProviders(<StandaloneHeader />);
    expect(screen.getByText("Chat")).toBeInTheDocument();
    expect(screen.getByText("Inspect")).toBeInTheDocument();
  });

  it("renders the view control segment group", () => {
    renderWithProviders(<StandaloneHeader />);
    expect(screen.getByTestId("view-control")).toBeInTheDocument();
  });

  it("does not render agent name when assistantId is null", () => {
    renderWithProviders(<StandaloneHeader />, {
      initialStoreState: { assistantId: null },
    });
    expect(screen.queryByText("finance-bot")).not.toBeInTheDocument();
  });

  it("renders agent name when assistantId is set", () => {
    renderWithProviders(<StandaloneHeader />, {
      initialStoreState: { assistantId: "finance-bot" },
    });
    expect(screen.getByText("finance-bot")).toBeInTheDocument();
  });

  it("clicking Inspect sets inspectMode to true", async () => {
    const user = userEvent.setup();
    renderWithProviders(<StandaloneHeader />, {
      initialStoreState: { inspectMode: false },
    });

    await user.click(screen.getByText("Inspect"));
    expect(inspectorStore.state.inspectMode).toBe(true);
  });

  it("clicking Chat sets inspectMode to false", async () => {
    const user = userEvent.setup();
    renderWithProviders(<StandaloneHeader />, {
      initialStoreState: { inspectMode: true },
    });

    await user.click(screen.getByText("Chat"));
    expect(inspectorStore.state.inspectMode).toBe(false);
  });
});
