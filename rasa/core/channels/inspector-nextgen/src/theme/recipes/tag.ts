import { defineSlotRecipe } from "@chakra-ui/react";

// TODO: add to ladle
export const tagSlotRecipe = defineSlotRecipe({
  slots: ["root", "label"],
  base: {
    root: {
      bg: "rasaNeutral.300",
      borderRadius: "0.5rem",
      paddingInline: "10px"
    },
    label: {
      fontFamily: "Soehne",
      fontWeight: "500",
    },
  },
  variants: {
    variant: {
      subtle: {
        root: {
          bg: "rasaNeutral.300",
        },
        label: {
          color: "rasaNeutral.800",
        },
      },
      surface: {
        root: {
          bg: "colorPalette.50",
          borderRadius: "0.5rem",
          borderWidth: "0",
          boxShadow: "none",
        },
        label: {
          color: "colorPalette.900",
        },
      }
    },
    size: {
      lg: {
        label: {
          fontSize: "12px",
        }
      }
    }
  },
});
