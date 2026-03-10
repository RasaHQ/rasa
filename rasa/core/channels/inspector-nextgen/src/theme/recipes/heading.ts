import { defineRecipe } from "@chakra-ui/react";

export const headingRecipe = defineRecipe({
  base: {
    fontWeight: "500",
    color: "rasawebDeepPurple.800",
    letterSpacing: "0.4px",
  },
  variants: {
    size: {
      sm: {
        fontSize: "13px",
      },
      md: {
        fontSize: "14px",
      },
      lg: {
        fontSize: "16px",
      },
      xxl: {
        fontSize: "42px",
      },
    },
  },
});
