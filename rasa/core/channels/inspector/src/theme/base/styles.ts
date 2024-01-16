import { mode, StyleFunctionProps } from "@chakra-ui/theme-tools";

export const styles = {
  global: {
    body: (theme: StyleFunctionProps) => ({
      bg: mode("neutral.300", "neutral.300")(theme),
      color: mode("neutral.900", "neutral.900")(theme),
      fontFamily: theme.fonts.body,
      fontSize: theme.rasaFontSizes.md,
      letterSpacing: "0.025rem",
    }),
  },
};
