import { createSystem, defaultConfig, defineConfig } from "@chakra-ui/react";

import { colors } from "./tokens/colors";
import { radii } from "./tokens/radii";
import { spacing } from "./tokens/spacing";
import { sizes } from "./tokens/sizes";
import { shadows } from "./tokens/shadows";
import { breakpoints } from "./tokens/breakpoints";
import { textRecipe } from "./recipes/text";
import { inputRecipe } from "./recipes/input";
import "../fonts.css";

const config = defineConfig({
  theme: {
    semanticTokens: {
      colors: {
        bg: { value: { base: "white", _dark: "black" } },
        "bg.subtle": { value: { base: "{colors.gray.50}", _dark: "{colors.gray.950}" } },
        "bg.muted": { value: { base: "{colors.gray.100}", _dark: "{colors.gray.900}" } },
        "bg.emphasized": { value: { base: "{colors.gray.200}", _dark: "{colors.gray.800}" } },
        "bg.inverted": { value: { base: "black", _dark: "white" } },
        "bg.panel": { value: { base: "white", _dark: "{colors.gray.950}" } },
        "bg.error": { value: { base: "{colors.red.50}", _dark: "{colors.red.950}" } },
        "bg.warning": { value: { base: "{colors.orange.50}", _dark: "{colors.orange.950}" } },
        "bg.success": { value: { base: "{colors.green.50}", _dark: "{colors.green.950}" } },
        "bg.info": { value: { base: "{colors.blue.50}", _dark: "{colors.blue.950}" } },

        border: { value: { base: "{colors.gray.200}", _dark: "{colors.gray.800}" } },
        "border.subtle": { value: { base: "{colors.gray.50}", _dark: "{colors.gray.950}" } },
        "border.muted": { value: { base: "{colors.gray.100}", _dark: "{colors.gray.900}" } },
        "border.emphasized": { value: { base: "{colors.gray.200}", _dark: "{colors.gray.700}" } },
        "border.inverted": { value: { base: "black", _dark: "white" } },
        "border.error": { value: { base: "{colors.red.600}", _dark: "{colors.red.500}" } },
        "border.warning": { value: { base: "{colors.orange.600}", _dark: "{colors.orange.500}" } },
        "border.success": { value: { base: "{colors.green.600}", _dark: "{colors.green.500}" } },
        "border.info": { value: { base: "{colors.blue.600}", _dark: "{colors.blue.500}" } },

        fg: { value: { base: "black", _dark: "white" } },
        "fg.muted": { value: { base: "{colors.gray.600}", _dark: "{colors.gray.400}" } },
        "fg.subtle": { value: { base: "{colors.gray.400}", _dark: "{colors.gray.600}" } },
        "fg.inverted": { value: { base: "{colors.gray.50}", _dark: "{colors.gray.900}" } },
        "fg.error": { value: { base: "{colors.red.500}", _dark: "{colors.red.400}" } },
        "fg.warning": { value: { base: "{colors.orange.600}", _dark: "{colors.orange.400}" } },
        "fg.success": { value: { base: "{colors.green.600}", _dark: "{colors.green.400}" } },
        "fg.info": { value: { base: "{colors.blue.600}", _dark: "{colors.blue.400}" } },

        "gray.contrast": { value: { base: "white", _dark: "black" } },
        "gray.fg": { value: { base: "{colors.gray.800}", _dark: "{colors.gray.200}" } },
        "gray.subtle": { value: { base: "{colors.gray.100}", _dark: "{colors.gray.900}" } },
        "gray.muted": { value: { base: "{colors.gray.200}", _dark: "{colors.gray.800}" } },
        "gray.emphasized": { value: { base: "{colors.gray.300}", _dark: "{colors.gray.700}" } },
        "gray.solid": { value: { base: "{colors.gray.900}", _dark: "{colors.gray.100}" } },
        "gray.focusRing": { value: { base: "{colors.gray.800}", _dark: "{colors.gray.200}" } },

        "purple.contrast": { value: { base: "white", _dark: "white" } },
        "purple.fg": { value: { base: "{colors.purple.700}", _dark: "{colors.purple.300}" } },
        "purple.subtle": { value: { base: "{colors.purple.100}", _dark: "{colors.purple.900}" } },
        "purple.muted": { value: { base: "{colors.purple.200}", _dark: "{colors.purple.800}" } },
        "purple.emphasized": { value: { base: "{colors.purple.300}", _dark: "{colors.purple.700}" } },
        "purple.solid": { value: { base: "{colors.purple.600}", _dark: "{colors.purple.500}" } },
        "purple.focusRing": { value: { base: "{colors.purple.600}", _dark: "{colors.purple.500}" } },

        "red.contrast": { value: { base: "white", _dark: "white" } },
        "red.fg": { value: { base: "{colors.red.700}", _dark: "{colors.red.300}" } },
        "red.subtle": { value: { base: "{colors.red.100}", _dark: "{colors.red.900}" } },
        "red.muted": { value: { base: "{colors.red.200}", _dark: "{colors.red.800}" } },
        "red.emphasized": { value: { base: "{colors.red.300}", _dark: "{colors.red.700}" } },
        "red.solid": { value: { base: "{colors.red.600}", _dark: "{colors.red.500}" } },
        "red.focusRing": { value: { base: "{colors.red.600}", _dark: "{colors.red.500}" } },

        "green.contrast": { value: { base: "white", _dark: "white" } },
        "green.fg": { value: { base: "{colors.green.700}", _dark: "{colors.green.300}" } },
        "green.subtle": { value: { base: "{colors.green.100}", _dark: "{colors.green.900}" } },
        "green.muted": { value: { base: "{colors.green.200}", _dark: "{colors.green.800}" } },
        "green.emphasized": { value: { base: "{colors.green.300}", _dark: "{colors.green.700}" } },
        "green.solid": { value: { base: "{colors.green.600}", _dark: "{colors.green.500}" } },
        "green.focusRing": { value: { base: "{colors.green.600}", _dark: "{colors.green.500}" } },

        "yellow.contrast": { value: { base: "black", _dark: "black" } },
        "yellow.fg": { value: { base: "{colors.yellow.700}", _dark: "{colors.yellow.300}" } },
        "yellow.subtle": { value: { base: "{colors.yellow.100}", _dark: "{colors.yellow.900}" } },
        "yellow.muted": { value: { base: "{colors.yellow.200}", _dark: "{colors.yellow.800}" } },
        "yellow.emphasized": { value: { base: "{colors.yellow.300}", _dark: "{colors.yellow.700}" } },
        "yellow.solid": { value: { base: "{colors.yellow.300}", _dark: "{colors.yellow.400}" } },
        "yellow.focusRing": { value: { base: "{colors.yellow.300}", _dark: "{colors.yellow.400}" } },

        "cyan.contrast": { value: { base: "white", _dark: "white" } },
        "cyan.fg": { value: { base: "{colors.cyan.700}", _dark: "{colors.cyan.300}" } },
        "cyan.subtle": { value: { base: "{colors.cyan.100}", _dark: "{colors.cyan.900}" } },
        "cyan.muted": { value: { base: "{colors.cyan.200}", _dark: "{colors.cyan.800}" } },
        "cyan.emphasized": { value: { base: "{colors.cyan.300}", _dark: "{colors.cyan.700}" } },
        "cyan.solid": { value: { base: "{colors.cyan.600}", _dark: "{colors.cyan.500}" } },
        "cyan.focusRing": { value: { base: "{colors.cyan.600}", _dark: "{colors.cyan.500}" } },

        "blue.contrast": { value: { base: "white", _dark: "white" } },
        "blue.fg": { value: { base: "{colors.blue.700}", _dark: "{colors.blue.300}" } },
        "blue.subtle": { value: { base: "{colors.blue.100}", _dark: "{colors.blue.900}" } },
        "blue.muted": { value: { base: "{colors.blue.200}", _dark: "{colors.blue.800}" } },
        "blue.emphasized": { value: { base: "{colors.blue.300}", _dark: "{colors.blue.700}" } },
        "blue.solid": { value: { base: "{colors.blue.600}", _dark: "{colors.blue.500}" } },
        "blue.focusRing": { value: { base: "{colors.blue.600}", _dark: "{colors.blue.500}" } },

        "teal.contrast": { value: { base: "white", _dark: "white" } },
        "teal.fg": { value: { base: "{colors.teal.700}", _dark: "{colors.teal.300}" } },
        "teal.subtle": { value: { base: "{colors.teal.100}", _dark: "{colors.teal.900}" } },
        "teal.muted": { value: { base: "{colors.teal.200}", _dark: "{colors.teal.800}" } },
        "teal.emphasized": { value: { base: "{colors.teal.300}", _dark: "{colors.teal.700}" } },
        "teal.solid": { value: { base: "{colors.teal.600}", _dark: "{colors.teal.500}" } },
        "teal.focusRing": { value: { base: "{colors.teal.600}", _dark: "{colors.teal.500}" } },

        "orange.contrast": { value: { base: "black", _dark: "black" } },
        "orange.fg": { value: { base: "{colors.orange.700}", _dark: "{colors.orange.300}" } },
        "orange.subtle": { value: { base: "{colors.orange.100}", _dark: "{colors.orange.900}" } },
        "orange.muted": { value: { base: "{colors.orange.200}", _dark: "{colors.orange.800}" } },
        "orange.emphasized": { value: { base: "{colors.orange.300}", _dark: "{colors.orange.700}" } },
        "orange.solid": { value: { base: "{colors.orange.600}", _dark: "{colors.orange.500}" } },
        "orange.focusRing": { value: { base: "{colors.orange.600}", _dark: "{colors.orange.500}" } },

        "pink.contrast": { value: { base: "white", _dark: "white" } },
        "pink.fg": { value: { base: "{colors.pink.700}", _dark: "{colors.pink.300}" } },
        "pink.subtle": { value: { base: "{colors.pink.100}", _dark: "{colors.pink.900}" } },
        "pink.muted": { value: { base: "{colors.pink.200}", _dark: "{colors.pink.800}" } },
        "pink.emphasized": { value: { base: "{colors.pink.300}", _dark: "{colors.pink.700}" } },
        "pink.solid": { value: { base: "{colors.pink.600}", _dark: "{colors.pink.500}" } },
        "pink.focusRing": { value: { base: "{colors.pink.600}", _dark: "{colors.pink.500}" } },
      },
    },
    tokens: {
      colors,
      radii,
      spacing,
      sizes,
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
        mono: {
          value:
            'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
        },
      },
      fontWeights: {
        normal: { value: 400 },
        medium: { value: 500 },
        bold: { value: 700 },
      },
      fontSizes: {
        xs: { value: "0.75rem" },
        sm: { value: "0.875rem" },
        md: { value: "1rem" },
        lg: { value: "1.125rem" },
        xl: { value: "1.25rem" },
        "2xl": { value: "1.5rem" },
        "3xl": { value: "1.875rem" },
        "4xl": { value: "2.25rem" },
        "5xl": { value: "3rem" },
        "6xl": { value: "3.75rem" },
        "7xl": { value: "4.5rem" },
      },
      lineHeights: {
        xs: { value: "1rem" },
        sm: { value: "1.25rem" },
        md: { value: "1.5rem" },
        lg: { value: "1.75rem" },
        xl: { value: "1.875rem" },
        "2xl": { value: "2rem" },
        "3xl": { value: "2.375rem" },
        "4xl": { value: "2.75rem" },
        "5xl": { value: "3.75rem" },
        "6xl": { value: "4.5rem" },
        "7xl": { value: "5.75rem" },
      },
      letterSpacings: {
        tight: { value: "-0.4px" },
      },
      animations: {
        blink: { value: "blink 1s ease-in-out infinite" },
      },
    },
    textStyles: {
      xs: {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.xs}",
          lineHeight: "{lineHeights.xs}",
        },
      },
      sm: {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.sm}",
          lineHeight: "{lineHeights.sm}",
        },
      },
      md: {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.md}",
          lineHeight: "{lineHeights.md}",
        },
      },
      lg: {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.lg}",
          lineHeight: "{lineHeights.lg}",
        },
      },
      xl: {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.xl}",
          lineHeight: "{lineHeights.xl}",
        },
      },
      "2xl": {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.2xl}",
          lineHeight: "{lineHeights.2xl}",
        },
      },
      "3xl": {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.3xl}",
          lineHeight: "{lineHeights.3xl}",
        },
      },
      "4xl": {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.4xl}",
          lineHeight: "{lineHeights.4xl}",
          letterSpacing: "{letterSpacings.tight}",
        },
      },
      "5xl": {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.5xl}",
          lineHeight: "{lineHeights.5xl}",
          letterSpacing: "{letterSpacings.tight}",
        },
      },
      "6xl": {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.6xl}",
          lineHeight: "{lineHeights.6xl}",
          letterSpacing: "{letterSpacings.tight}",
        },
      },
      "7xl": {
        value: {
          fontFamily: "{fonts.body}",
          fontSize: "{fontSizes.7xl}",
          lineHeight: "{lineHeights.7xl}",
          letterSpacing: "{letterSpacings.tight}",
        },
      },
    },
    recipes: {
      text: textRecipe,
      input: inputRecipe,
    },
    breakpoints,
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
