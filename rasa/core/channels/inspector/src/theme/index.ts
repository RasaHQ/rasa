import {
  extendTheme,
  withDefaultColorScheme,
  useTheme,
  ChakraTheme,
  type ThemeConfig,
} from '@chakra-ui/react'
import { breakpoints, Breakpoints } from './base/breakpoints'
import { rasaColors, RasaColors } from './base/colors'
import { rasaRadii, RasaRadii } from './base/radii'
import { rasaShadows, RasaShadows } from './base/shadows'
import { rasaSizes, RasaSizes } from './base/sizes'
import { rasaSpace, RasaSpace } from './base/space'
import { rasaZIndices, RasaZIndices } from './base/zIndices'
import { styles } from './base/styles'
import {
  fonts,
  rasaFontSizes,
  rasaFontWeights,
  RasaFontSizes,
  RasaFontWeights,
} from './base/typography'

// Custom components
import { Button } from './Button/Button'
import { Heading } from './Heading/Heading'
import { Input } from './Input/Input'
import { Link } from './Link/Link'
import { Table } from './Table/Table'
import { Modal } from './Modal/Modal'
import { Tooltip } from './Tooltip/Tooltip'

export interface CustomTheme {
  // default types
  config: ThemeConfig
  semanticTokens: ChakraTheme['semanticTokens']
  direction: ChakraTheme['direction']
  transition: ChakraTheme['transition']

  // we merge & override
  styles: ChakraTheme['styles']
  fonts: RasaFontSizes
  breakpoints: Breakpoints

  // we also merge & override, but use custom color names
  colors: RasaColors

  // custom key & types
  rasaFontSizes: RasaFontSizes
  rasaFontWeights: RasaFontWeights
  rasaRadii: RasaRadii
  rasaSpace: RasaSpace
  rasaShadows: RasaShadows
  rasaSizes: RasaSizes
  zIndices: ChakraTheme['zIndices'] & RasaZIndices
}

const config: ThemeConfig = {
  initialColorMode: 'light',
  useSystemColorMode: false,
}

// Extend theme deep merges with the default Chakra theme.
// We try to keep many defaults in tact to support
// - default Chakra components
// - external Chakra components like chakra-react-select.
//
// But we also provide our own custom keys / values, that can be tuned to our liking.
export const theme = extendTheme(
  {
    config,
    components: {
      Button,
      Heading,
      Input,
      Link,
      Modal,
      Table,
      Tooltip,
    },

    styles,
    fonts,

    breakpoints,
    colors: rasaColors,

    rasaFontSizes,
    rasaFontWeights,
    rasaRadii,
    rasaSpace,
    rasaShadows,
    rasaSizes,
    zIndices: rasaZIndices,
  },
  withDefaultColorScheme({ colorScheme: 'rasaPurple' }),
)

export const useOurTheme = () => {
  return useTheme<CustomTheme>()
}
