import {
  Box,
  FlexProps,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  Flex,
  useColorModeValue,
  Tooltip,
} from '@chakra-ui/react'
import { useOurTheme } from '../theme'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import { Slot } from '../types'
import { shouldShowTooltip } from '../helpers/utils'

interface Props extends FlexProps {
  slots: Slot[]
}

export const SlotTable = ({ slots }: { slots: Slot[] }) => {
  const { rasaFontSizes } = useOurTheme()

  const highlighterSx = {
    background: useColorModeValue('neutral.50', 'neutral.50'),
    fontSize: rasaFontSizes.sm,
    letterSpacing: '0',
  }

  return (
    <Table width="100%" layout="fixed">
      <Thead>
        <Tr>
          <Th w="50%">Name</Th>
          <Th>Value</Th>
        </Tr>
      </Thead>
      <Tbody>
        {slots.map((slot) => (
          <Tr key={slot.name}>
            <Td>
              {shouldShowTooltip(slot.name) ? (
                <Tooltip label={slot.name} hasArrow>
                  <Text noOfLines={1}>{slot.name}</Text>
                </Tooltip>
              ) : (
                <Text noOfLines={1}>{slot.name}</Text>
              )}
            </Td>
            <Td>
              <SyntaxHighlighter customStyle={highlighterSx}>
                {JSON.stringify(slot.value, null, 2)}
              </SyntaxHighlighter>
            </Td>
          </Tr>
        ))}
      </Tbody>
    </Table>
  )
}

export const Slots = ({ sx, slots, ...props }: Props) => {
  const { rasaSpace } = useOurTheme()

  const containerSx = {
    ...sx,
    pr: 0,
    pb: 0,
    position: 'relative',
    flexDirection: 'column',
  }
  const overflowBox = {
    height: '100%',
    overflow: 'auto',
    pr: rasaSpace[1],
    pb: rasaSpace[0.5],
  }

  const displaySlots = slots.length ? slots : [{ name: '-', value: '-' }]

  return (
    <Flex sx={containerSx} {...props}>
      <Box sx={overflowBox}>
        <SlotTable slots={displaySlots} />
      </Box>
    </Flex>
  )
}
