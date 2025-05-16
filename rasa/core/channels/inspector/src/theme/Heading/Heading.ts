import { StyleConfig } from '@chakra-ui/theme-tools'

export const Heading: StyleConfig = {
  baseStyle: {
    fontWeight: 'bold',
    letterSpacing: 'wide',
  },
  sizes: {
    sm: ({ theme }) => ({
      fontSize: theme.rasaFontSizes.sm,
      lineHeight: 'short',
      margin: 0,
    }),
    md: ({ theme }) => ({
      fontSize: theme.rasaFontSizes.md,
      margin: 0,
    }),
    lg: ({ theme }) => ({
      fontSize: theme.rasaFontSizes.lg,
      margin: 0,
    }),
    xl: ({ theme }) => ({
      fontSize: theme.rasaFontSizes.xl,
      lineHeight: '100%',
      margin: 0,
    }),
  },
  defaultProps: {
    size: 'md',
  },
}
