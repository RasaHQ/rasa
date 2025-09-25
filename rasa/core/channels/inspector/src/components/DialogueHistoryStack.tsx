import {
  Box,
  FlexProps,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  useColorModeValue,
  Flex,
  Tooltip,
} from '@chakra-ui/react'
import { useOurTheme } from '../theme'
import { Stack } from '../types'
import { shouldShowTooltip } from '../helpers/utils'

interface Props extends FlexProps {
  stack: Stack[]
  active?: Stack
  onItemClick?: (stack: Stack) => void
}

function StackRow({
  stack,
  highlighted,
  selectable,
  onItemClick,
}: {
  stack: Stack
  highlighted?: boolean
  selectable?: boolean
  onItemClick?: (stack: Stack) => void
}) {
  // use pointy hand cursor when hovering over a row
  const clickableTrSx = {
    _hover: {
      cursor: 'pointer',
    },
  }

  const highlightedTrSx = {
    td: {
      bg: useColorModeValue('warning.50', 'warning.50'),
    },
    _last: {
      td: {
        border: 'none',
      },
    },
  }

  return (
    <Tr
      sx={
        highlighted ? highlightedTrSx : selectable ? clickableTrSx : undefined
      }
      onClick={() => onItemClick?.(stack)}
    >
      <Td>
        <Tooltip label={`${stack.flow_id} (${stack.frame_id})`} hasArrow>
          <Text noOfLines={1}>{stack.flow_id}</Text>
        </Tooltip>
      </Td>
      <Td>
        {shouldShowTooltip(stack.step_id) ? (
          <Tooltip label={stack.step_id} hasArrow>
            <Text noOfLines={1}>{stack.step_id}</Text>
          </Tooltip>
        ) : (
          <Text noOfLines={1}>{stack.step_id}</Text>
        )}
      </Td>
    </Tr>
  )
}

export const DialogueHistoryStack = ({
  sx,
  stack,
  active,
  onItemClick,
  ...props
}: Props) => {
  const { rasaSpace } = useOurTheme()

  const containerSx = {
    ...sx,
    pr: 0,
    pb: 0,
    flexDirection: 'column',
  }
  const overflowBox = {
    height: '100%',
    overflow: 'auto',
    pr: rasaSpace[1],
    pb: rasaSpace[0.5],
  }

  return (
    <Flex sx={containerSx} {...props}>
      <Flex>
        <Heading size="lg" mb={rasaSpace[0.5]}>
          History
        </Heading>
        <Text ml={rasaSpace[0.25]}>({stack.length} {stack.length === 1 ? 'flow' : 'flows'})</Text>
      </Flex>
      <Box sx={overflowBox}>
        <Table width="100%" layout="fixed">
          <Thead>
            <Tr>
              <Th>Flow</Th>
              <Th width="40%">Step ID</Th>
            </Tr>
          </Thead>
          <Tbody>
            {stack.length > 0 &&
              [...stack]
                .filter(frame => !frame.agent_id)
                .reverse()
                .map((stack) => (
                  <StackRow
                    stack={stack}
                    highlighted={stack.frame_id == active?.frame_id}
                    onItemClick={onItemClick}
                    selectable={true}
                  />
                ))}
            {stack.length === 0 && (
              <StackRow
                stack={{
                  frame_id: '-',
                  flow_id: '-',
                  step_id: '-',
                  ended: false,
                  type: "flow"
                }}
              />
            )}
          </Tbody>
        </Table>
      </Box>
    </Flex>
  )
}
