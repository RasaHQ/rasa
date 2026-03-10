import { createSystem, defaultConfig, defineConfig } from "@chakra-ui/react";

import { colors } from "./tokens/colors";
import { dialogSlotRecipe } from "./recipes/dialog";
import { spacing } from "./tokens/spacing";
import { shadows } from "./tokens/shadows";
import { buttonRecipe } from "./recipes/button";
import "../fonts.css";
import { headingRecipe } from "./recipes/heading";
import { textRecipe } from "./recipes/text";
import { segmentGroupSlotRecipe } from "./recipes/segmentGroup";
import { tagSlotRecipe } from "./recipes/tag";
import { toasterSlotRecipe } from "./recipes/toaster";

const config = defineConfig({
  theme: {
    tokens: {
      colors,
      spacing,
      shadows,
      fonts: {
        heading: {
          value:
            'Soehne, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        },
        body: {
          value:
            'Soehne, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        },
      },
      fontWeights: {
        normal: { value: 400 },
        medium: { value: 500 },
      },
      animations: {
        blink: { value: "blink 1s ease-in-out infinite" },
      },
    },
    textStyles: {
      heading: {
        value: {
          fontFamily: "{fonts.heading}",
          fontWeight: "{fontWeights.medium}",
        },
      },
      body: {
        value: {
          fontFamily: "{fonts.body}",
          fontWeight: "{fontWeights.normal}",
        },
      },
    },
    semanticTokens: {
      colors: {
        // Color palettes
        purple: {
          solid: { value: "{colors.rasawebPurple.800}" },
          fg: { value: "{colors.rasawebPurple.900}" },
          contrast: { value: "{colors.rasaNeutral.50}" },
          subtle: { value: "{colors.rasawebPurple.50}" },
          muted: { value: "{colors.rasaNeutral.200}" },
          emphasized: { value: "{colors.rasaNeutral.500}" },
        },
        dark: {
          solid: { value: "{colors.rasawebDeepPurple.800}" },
          fg: { value: "{colors.rasawebDeepPurple.900}" },
          contrast: { value: "{colors.rasaNeutral.50}" },
          subtle: { value: "{colors.rasaNeutral.700}" },
          muted: { value: "{colors.rasaNeutral.200}" },
          emphasized: { value: "{colors.rasaNeutral.500}" },
        },
        light: {
          solid: { value: "{colors.rasaNeutral.50}" },
          fg: { value: "{colors.rasaNeutral.300}" },
          contrast: { value: "{colors.rasaNeutral.700}" },
        },
        copilot: {
          solid: { value: "{colors.rasawebDeepPurple.800}" },
          fg: { value: "{colors.rasawebDeepPurple.800}" },
          contrast: { value: "{colors.rasaNeutral.50}" },
          subtle: { value: "{colors.rasaNeutral.700}" },
          muted: { value: "{colors.rasaNeutral.200}" },
          emphasized: { value: "{colors.rasaNeutral.600}" },
        },
      },
    },
    recipes: {
      button: buttonRecipe,
      iconButton: buttonRecipe,
      heading: headingRecipe,
      text: textRecipe,
    },
    slotRecipes: {
      segmentGroup: segmentGroupSlotRecipe,
      tag: tagSlotRecipe,
      toast: toasterSlotRecipe,
      dialog: dialogSlotRecipe,
    },
    breakpoints: {
      sm: "480px",
      md: "768px",
      lg: "1024px",
      xl: "1280px",
      "2xl": "1536px",
      "3xl": "1920px",
    },
    keyframes: {
      blink: {
        "0%, 100%": {},
        "50%": { backgroundColor: "#6B7694", transform: "scale(1.3)" },
      },
    },
  },
  cssVarsPrefix: "app",
  preflight: true,
});

export const system = createSystem(defaultConfig, config);
