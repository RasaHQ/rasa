import { defineRecipe } from "@chakra-ui/react";

export const textRecipe = defineRecipe({
  base: {
    fontWeight: "400",
    color: "rasawebDeepPurple.800",
    letterSpacing: "0.4px",
  },
  variants: {
    size: {
      xs: {
        fontSize: "11px",
        lineHeight: "16px",
      },
      sm: {
        fontSize: "13px",
        lineHeight: "20px",
      },
      md: {
        fontSize: "14px",
        lineHeight: "20px",
      },
      xxl: {
        fontSize: "20px",
        lineHeight: "28px",
      },
    },
    variant: {
      primary: {
        color: "rasawebDeepPurple.800",
      },
      muted: {
        color: "rasaNeutral.700",
      },
    },
  },
  defaultVariants: {
    variant: "primary",
  },
});
