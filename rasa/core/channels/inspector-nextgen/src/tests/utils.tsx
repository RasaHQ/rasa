import { ChakraProvider, defaultSystem } from "@chakra-ui/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  type RenderOptions,
  type RenderResult,
  render,
} from "@testing-library/react";
import React from "react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { initInspectorStore, type InspectorStoreState } from "../store";

export function renderWithProviders(
  children: React.ReactElement,
  options?: Omit<RenderOptions, "queries"> & {
    initialStoreState?: Partial<InspectorStoreState>;
  },
): RenderResult {
  const queryClient = new QueryClient();
  const { initialStoreState, ...renderOptions } = options ?? {};

  initInspectorStore(initialStoreState);

  const Providers = ({ children }: { children: React.ReactNode }) => {
    return (
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>
          <ChakraProvider value={defaultSystem}>{children}</ChakraProvider>
        </QueryClientProvider>
      </MemoryRouter>
    );
  };

  return render(children, { wrapper: Providers, ...renderOptions });
}

export const defaultAnalyticsMock = {
  track: vi.fn(),
  identify: vi.fn(),
  page: vi.fn(),
  identifyUser: vi.fn(),
  getSegmentUserId: vi.fn(),
  TRACKING_EVENTS: new Proxy(
    {},
    {
      get: () => "mocked_event",
    },
  ),
  UTM_PARAMS: {
    utm_source: "hello_rasa",
    utm_medium: "connect_popup",
    utm_campaign: "community_growth",
    utm_content: "join_community",
  },
};
