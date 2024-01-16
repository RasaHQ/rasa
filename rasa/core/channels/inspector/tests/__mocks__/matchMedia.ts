import { jest } from "@jest/globals";

// Mock window.matchMedia to allow Chakra useBreakpointValue to work
Object.defineProperty(window, "matchMedia", {
  value: () => {
    return {
      matches: false,
      onchange: null,
      addListener: jest.fn(), // Deprecated
      removeListener: jest.fn(), // Deprecated
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    };
  },
});
