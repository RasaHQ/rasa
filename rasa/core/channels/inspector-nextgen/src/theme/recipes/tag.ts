import { defineSlotRecipe } from "@chakra-ui/react";

// TODO: add to ladle
export const tagSlotRecipe = defineSlotRecipe({
  slots: ["root", "label"],
  base: {
    root: {
      bg: "colorPalette.subtle",
      borderRadius: "0.5rem",
    },
    label: {
      color: "colorPalette.solid",
      fontWeight: "500",
      fontSize: "13px",
    },
  },
  variants: {},
});
