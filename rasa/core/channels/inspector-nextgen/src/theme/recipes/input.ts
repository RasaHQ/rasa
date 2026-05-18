import { defineRecipe } from "@chakra-ui/react";

export const inputRecipe = defineRecipe({
  variants: {
    size: {
      "3xl": {
        h: "16",
        px: "6",
        borderRadius: "full",
        fontSize: "sm",
      },
    },
  },
});
