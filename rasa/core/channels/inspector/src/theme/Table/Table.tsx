import { mode, StyleFunctionProps } from '@chakra-ui/theme-tools'
import type { ComponentStyleConfig } from '@chakra-ui/theme'

export const Table: ComponentStyleConfig = {
  baseStyle: (props: StyleFunctionProps) => ({
    table: {
      thead: {
        bgColor: mode('neutral.50', 'neutral.50')(props),
        position: 'sticky',
        top: 0,
        zIndex: props.theme.zIndices.docked,
      },
      th: {
        color: mode('neutral.900', 'neutral.900')(props),
        fontSize: props.theme.rasaFontSizes.md,
        py: props.theme.rasaSpace[0.5],
        pl: 0,
        pr: props.theme.rasaSpace[0.5],
        border: 'none',
        textTransform: 'none',
      },
      tr: {
        td: {
          borderColor: props.theme.colors.neutral[400],
        },
        _last: {
          td: { borderBottom: 'none' },
        },
      },
      td: {
        py: props.theme.rasaSpace[0.5],
        pl: 0,
        pr: props.theme.rasaSpace[0.5],
        bgColor: mode('neutral.50', 'neutral.50')(props),
      },
    },
  }),
}
