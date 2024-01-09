import React from "react";
import { render, RenderResult, RenderOptions } from "@testing-library/react";
import { ChakraProvider } from "@chakra-ui/react";
import { theme } from "../src/theme";

export function renderWithProviders(
  children: React.ReactElement,
  options?: Omit<RenderOptions, "queries">
): RenderResult {
  return render(
    <ChakraProvider theme={theme}>{children}</ChakraProvider>,
    options
  );
}
