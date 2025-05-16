import type { ComponentStyleConfig } from '@chakra-ui/theme'
import { mode } from '@chakra-ui/theme-tools'

export const Button: ComponentStyleConfig = {
  defaultProps: {
    variant: 'solidRasa',
  },
  baseStyle: () => ({
    letterSpacing: 'wide',
  }),
  sizes: {
    md: ({ theme }) => ({
      fontSize: theme.rasaFontSizes.md,
      borderRadius: theme.rasaRadii.normal,
    }),
  },
  variants: {
    solidRasa: (props) => {
      const { theme } = props
      return {
        ...theme.components.Button.variants.solid(props),
        backgroundColor: mode(
          theme.colors.rasaPurple[800],
          theme.colors.rasaPurple[800],
        )(props),
      }
    },
  },
}
