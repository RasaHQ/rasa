import { defineSlotRecipe } from "@chakra-ui/react";
import { accordionAnatomy } from "@chakra-ui/react/anatomy";

// TODO: support colorPalettes
export const accordionSlotRecipe = defineSlotRecipe({
  slots: accordionAnatomy.keys(),
  base: {
    itemIndicator: {
      color: "rasaNeutral.700",
    }
  }
});
