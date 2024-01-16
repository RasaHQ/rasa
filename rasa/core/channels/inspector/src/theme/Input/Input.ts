import { StyleConfig } from "@chakra-ui/theme-tools";
import { mode } from "@chakra-ui/theme-tools";

export const Input: StyleConfig = {
  baseStyle: (props) => ({
    field: {
      "::-webkit-calendar-picker-indicator": {
        display: "none",
      },
      ":disabled": {
        cursor: "not-allowed",
        opacity: "1 !important",
        backgroundColor: mode(
          props.theme.colors.neutral[300],
          props.theme.colors.neutral[300]
        )(props),
      },
    },
  }),
  sizes: {
    md: ({ theme }) => ({
      field: {
        fontSize: theme.rasaFontSizes.md,
      },
    }),
  },
};
