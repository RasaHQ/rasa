import { defineSlotRecipe } from "@chakra-ui/react";

export const dialogSlotRecipe = defineSlotRecipe({
  slots: [
    "root",
    "backdrop",
    "positioner",
    "content",
    "header",
    "body",
    "footer",
    "closeTrigger",
    "trigger",
  ],
  base: {
    content: {
      borderRadius: "1rem",
    },
  },
  variants: {
    size: {
      sm: {
        content: {
          maxW: "448px",
        },
      },
      md: {
        content: {
          maxW: "684px",
        },
      },
      lg: {
        content: {
          maxW: "894px",
        },
      },
    },
  },
  defaultVariants: {
    size: "sm",
  },
});
