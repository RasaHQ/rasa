import { StyleConfig } from '@chakra-ui/theme-tools'

export const Tooltip: StyleConfig = {
  baseStyle: ({ theme }) => ({
    letterSpacing: 'wide',
    borderRadius: theme.rasaRadii.normal,
    padding: theme.rasaSpace[0.5],
    textAlign: 'center',
    bgColor: theme.colors.neutral[800],
    '--tooltip-bg': theme.colors.neutral[800],
  }),
}
