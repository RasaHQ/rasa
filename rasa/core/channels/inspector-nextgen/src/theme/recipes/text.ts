import { defineRecipe } from "@chakra-ui/react";

export const textRecipe = defineRecipe({
  variants: {
    fontFamily: {
      sans: { fontFamily: "{fonts.body}" },
      mono: { fontFamily: "{fonts.mono}" },
    },
  },
});
