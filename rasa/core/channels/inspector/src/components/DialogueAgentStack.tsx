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
  Flex,
} from '@chakra-ui/react'
import { useOurTheme } from '../theme'
import {Agent, Stack} from '../types'

function mapStatusToHumanReadableName(status: Agent['status']) {
  switch (status) {
    case 'running':
      return 'Running'
    case 'completed':
      return 'Completed'
    case 'interrupted':
      return 'Interrupted'
    case 'cancelled':
      return 'Cancelled'
    default:
      return '-'
  }
}

interface Props extends FlexProps {
  agents: Agent[]
  active?: Stack
  onItemClick?: (stack: Stack) => void
}

function AgentRow({
  agent,
}: {
  agent: Agent
}) {

  return (
    <Tr>
      <Td>
          <Text noOfLines={1}>{agent.name}</Text>
      </Td>
      <Td>
          <Text noOfLines={1}>{mapStatusToHumanReadableName(agent.status)}</Text>
      </Td>
    </Tr>
  )
}

export const DialogueAgentStack = ({
  sx,
  agents,
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
          Agents
        </Heading>
        <Text ml={rasaSpace[0.25]}>({agents.length} {agents.length === 1 ? 'agent' : 'agents'})</Text>
      </Flex>
      <Box sx={overflowBox}>
        <Table width="100%" layout="fixed">
          <Thead>
            <Tr>
              <Th>Name</Th>
              <Th width="40%">Status</Th>
            </Tr>
          </Thead>
          <Tbody>
            {agents.length > 0 &&
              [...agents]
                .reverse()
                .map((agent) => (
                  <AgentRow
                    agent={agent}
                  />
                ))}
          </Tbody>
        </Table>
      </Box>
    </Flex>
  )
}
