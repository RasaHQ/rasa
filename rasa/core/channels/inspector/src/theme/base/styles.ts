import {mode, StyleFunctionProps} from '@chakra-ui/theme-tools'
import {keyframes} from "@chakra-ui/react";


const pulse = keyframes`
    0% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
    100% {
        opacity: 1;
    }
`;


export const styles = {
  global: {
    body: (theme: StyleFunctionProps) => ({
      bg: mode('neutral.300', 'neutral.300')(theme),
      color: mode('neutral.900', 'neutral.900')(theme),
      fontFamily: theme.fonts.body,
      fontSize: theme.rasaFontSizes.md,
      letterSpacing: '0.025rem',
    }),
    '.pulse': {
      animation: `${pulse} 2s infinite`,
    },
  },
}
