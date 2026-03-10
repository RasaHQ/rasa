import { defineSlotRecipe } from "@chakra-ui/react";

// TODO: support colorPalettes
export const segmentGroupSlotRecipe = defineSlotRecipe({
  slots: ["root", "item", "itemText", "indicator", "label"],
  base: {
    root: {
      bg: "rasaNeutral.50",
      height: "2rem",
      borderRadius: "0.75rem",
      color: "rasaNeutral.400", // for the border/boxShadow
      boxShadow: "inset 0 0 0 1px",
      gap: "0.25rem",
      p: "1",
      divideStyle: "none",
      "& label": {
        padding: "0.6rem",
        height: "1.5rem",
        lineHeight: "1.5rem",
        borderRadius: "0.5rem",
        "&:not(:hover)": {
          backgroundColor: "transparent !important", // fix for chakra ui buggy white bg
        },
        "&:hover:not([data-state='checked'])": {
          bg: "rasaNeutral.300",
          color: "rasawebDeepPurple.800",
        },
        "&:hover[data-state='checked']": {
          bg: "rasawebDeepPurple.900",
        },
      },
    },
    item: {
      fontWeight: "500",
      color: "rasaNeutral.700",
      lineHeight: "1.5rem",
      cursor: "pointer",
    },
    indicator: {
      bg: "rasawebDeepPurple.800",
      color: "rasaNeutral.50",
      height: "1.5rem",
      borderRadius: "0.5rem",
    },
    itemText: {
      height: "1.5rem",
      fontSize: "13px",
      "&[data-state='checked']": {
        color: "rasaNeutral.50",
      },
    },
  },
  variants: {},
});
