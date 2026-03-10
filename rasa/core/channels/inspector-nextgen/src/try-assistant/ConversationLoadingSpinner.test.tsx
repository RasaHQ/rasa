import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../tests/utils";
import { ConversationLoadingSpinner } from "./ConversationLoadingSpinner";

describe("ConversationLoadingSpinner", () => {
  it("renders three loading dots inside the MessageMarkup", () => {
    renderWithProviders(<ConversationLoadingSpinner />);
    const spinner = screen.getByTestId("agent-loading-spinner");
    expect(spinner).toBeInTheDocument();

    // The Flex is child of MessageMarkup, which contains the 3 Box dots
    const dots = screen.getByTestId("loading-dots").children;
    expect(dots).toHaveLength(3);
  });

  it("has MessageMarkup with the agent-loading-spinner test id", () => {
    renderWithProviders(<ConversationLoadingSpinner />);
    const spinner = screen.getByLabelText("Agent is typing");
    expect(spinner).toBeInTheDocument();
  });

  it("renders without crashing", () => {
    expect(() =>
      renderWithProviders(<ConversationLoadingSpinner />),
    ).not.toThrow();
  });
});

