import "@testing-library/jest-dom";
import { vi } from "vitest";
import ResizeObserver from "resize-observer-polyfill";
import { defaultAnalyticsMock } from "./src/tests/utils.tsx";

globalThis.ResizeObserver = ResizeObserver;
// Global analytics mock for all tests
vi.mock("./src/services/analytics", () => defaultAnalyticsMock);

