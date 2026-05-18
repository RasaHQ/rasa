import { Status } from "@chakra-ui/react";

export const ErrorDot = () => (
  <Status.Root colorPalette="red" ml="2" data-testid="error-dot">
    <Status.Indicator />
  </Status.Root>
);
