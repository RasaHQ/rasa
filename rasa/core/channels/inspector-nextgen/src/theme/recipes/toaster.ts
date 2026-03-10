import { defineSlotRecipe } from "@chakra-ui/react";

export const toasterSlotRecipe = defineSlotRecipe({
  slots: ["root"],
  base: {
    root: {
      borderRadius: "2xl",
      "&[data-type=warning]": {
        bg: "rasawebPurple.800",
        color: "rasaNeutral.50",
      },
    },
  },
});
