import "@testing-library/jest-dom";
import { vi } from "vitest";
import ResizeObserver from "resize-observer-polyfill";
import { defaultAnalyticsMock } from "./src/tests/utils.tsx";

globalThis.ResizeObserver = ResizeObserver;

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Global analytics mock for all tests
vi.mock("./src/services/analytics", () => defaultAnalyticsMock);

