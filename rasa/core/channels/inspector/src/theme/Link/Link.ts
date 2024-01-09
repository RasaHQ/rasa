import type { ComponentStyleConfig } from "@chakra-ui/theme";
import { mode } from "@chakra-ui/theme-tools";

export const Link: ComponentStyleConfig = {
  baseStyle: (props) => ({
    color: mode("rasaPurple.700", "rasaPurple.700")(props),
    fontWeight: "bold",
    textDecoration: "underline",
  }),
};
