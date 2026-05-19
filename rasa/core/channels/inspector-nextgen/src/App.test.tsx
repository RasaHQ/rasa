import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, render, screen } from "@testing-library/react";
import App from "./App";

// Capture the callback Inspector receives so we can invoke it in tests
let capturedOnInspectModeChange: ((inspect: boolean) => void) | undefined;

vi.mock("./Inspector", () => ({
  Inspector: (props: {
    projectUrl: string;
    botDataEndpoint: string;
    singleSessionMode: boolean;
    onInspectModeChange: (inspect: boolean) => void;
  }) => {
    capturedOnInspectModeChange = props.onInspectModeChange;
    return (
      <div
        data-testid="inspector"
        data-project-url={props.projectUrl}
        data-endpoint={props.botDataEndpoint}
        data-single-session={String(props.singleSessionMode)}
      />
    );
  },
}));

vi.mock("./components/StandaloneHeader", () => ({
  StandaloneHeader: () => <div data-testid="standalone-header" />,
}));

vi.mock("./assets/images", () => ({
  Background: "mock-background.png",
}));

vi.mock("react-router-dom", () => ({
  useSearchParams: () => [{ get: () => null }],
}));

// Use defaultSystem so ChakraProvider gets a valid theme in jsdom
vi.mock("./theme", async () => {
  const { defaultSystem } = await import("@chakra-ui/react");
  return { system: defaultSystem };
});

describe("App", () => {
  beforeEach(() => {
    capturedOnInspectModeChange = undefined;
  });

  it("renders without crashing", () => {
    render(<App />);
  });

  it("renders StandaloneHeader", () => {
    render(<App />);
    expect(screen.getByTestId("standalone-header")).toBeInTheDocument();
  });

  it("renders Inspector", () => {
    render(<App />);
    expect(screen.getByTestId("inspector")).toBeInTheDocument();
  });

  it("passes /data as botDataEndpoint to Inspector", () => {
    render(<App />);
    expect(screen.getByTestId("inspector")).toHaveAttribute("data-endpoint", "/data");
  });

  it("passes singleSessionMode=true to Inspector", () => {
    render(<App />);
    expect(screen.getByTestId("inspector")).toHaveAttribute("data-single-session", "true");
  });

  describe("projectUrl", () => {
    it("uses http://localhost:5005 when running in dev mode", () => {
      // Vitest runs with import.meta.env.DEV = true, matching the dev branch
      render(<App />);
      expect(screen.getByTestId("inspector")).toHaveAttribute(
        "data-project-url",
        "http://localhost:5005",
      );
    });

    it("uses globalThis.location.origin when running in production mode", () => {
      // Override import.meta.env.DEV to simulate a production build
      const env = import.meta.env as Record<string, unknown>;
      const original = env.DEV;
      env.DEV = false;

      try {
        render(<App />);
        expect(screen.getByTestId("inspector")).toHaveAttribute(
          "data-project-url",
          globalThis.location.origin,
        );
      } finally {
        env.DEV = original;
      }
    });
  });

  describe("inspect mode container width", () => {
    it("passes an onInspectModeChange callback to Inspector", () => {
      render(<App />);
      expect(capturedOnInspectModeChange).toBeTypeOf("function");
    });

    it("re-renders without errors when inspect mode is enabled", () => {
      render(<App />);
      act(() => capturedOnInspectModeChange!(true));
      expect(screen.getByTestId("inspector")).toBeInTheDocument();
    });

    it("re-renders without errors when inspect mode is disabled after being enabled", () => {
      render(<App />);
      act(() => capturedOnInspectModeChange!(true));
      act(() => capturedOnInspectModeChange!(false));
      expect(screen.getByTestId("inspector")).toBeInTheDocument();
    });
  });
});
