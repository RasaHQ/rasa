import { defineRecipe } from "@chakra-ui/react";

export const buttonRecipe = defineRecipe({
  base: {
    _disabled: {
      opacity: "0.4",
    },
    borderRadius: "lg",
  },
  variants: {
    size: {
      sm: {
        height: "8",
        minWidth: "2rem",
      },
    },
    variant: {
      solid: {
        bg: "colorPalette.solid",
        color: "colorPalette.contrast",
        _hover: {
          bg: "colorPalette.fg",
        },
        _expanded: {
          bg: "colorPalette.fg",
        },
        _disabled: {
          bg: "colorPalette.solid",
        },
      },
      outline: {
        bg: "colorPalette.contrast",
        color: "colorPalette.solid",
        borderColor: "colorPalette.solid",
        _hover: {
          color: "colorPalette.fg",
          bg: "colorPalette.muted",
        },
        _expanded: {
          color: "colorPalette.emphasized",
          bg: "colorPalette.muted",
        },
        _disabled: {
          color: "colorPalette.fg",
        },
      },
      // No color palette support for ghost variant
      ghost: {
        bg: "transparent",
        color: "rasawebDeepPurple.800",
        _hover: {
          color: "rasawebPurple.900",
          bg: "transparent",
        },
        _disabled: {
          color: "rasawebDeepPurple.800",
        },
      },
      subtle: {
        bg: "colorPalette.contrast",
        color: "colorPalette.subtle",
        _hover: {
          color: "colorPalette.fg",
          bg: "none",
        },
        _disabled: {
          color: "colorPalette.fg",
        },
      },
    },
  },
});
