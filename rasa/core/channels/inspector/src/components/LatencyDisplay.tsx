import {
  Box,
  Flex,
  FlexProps,
  Table,
  Tbody,
  Td,
  Text,
  Tooltip,
  Tr,
  useColorModeValue,
} from '@chakra-ui/react'
import { useOurTheme } from '../theme'
import {
  isRasaLatency,
  isVoiceLatency,
  RasaLatency,
  VoiceLatency,
} from '../types'

interface MinimalDisplayProps extends FlexProps {
  latency: RasaLatency
}

/**
 * Simple latency display for text-only conversations.
 * Shows a single response time value.
 */
const MinimalDisplay = ({ latency, sx, ...props }: MinimalDisplayProps) => {
  const containerSx = {
    ...sx,
    display: 'flex',
    alignItems: 'center',
  }

  const getLatencyColor = (latency: number) => {
    if (latency < 1500) return 'green.500'
    if (latency < 2500) return 'orange.500'
    return 'red.500'
  }

  const value = Math.round(
    (latency.command_processor || 0) + (latency.prediction_loop || 0),
  )
  const color = getLatencyColor(value)

  return (
    <Flex sx={containerSx} {...props}>
      <Text fontSize="md" color={useColorModeValue('gray.700', 'gray.300')}>
        Response latency:
        <Text as="span" ml={2} fontWeight="bold" color={color}>
          {value}ms
        </Text>
      </Text>
    </Flex>
  )
}

interface WaterfallDisplayProps extends FlexProps {
  latency: VoiceLatency
}

/**
 * Detailed latency waterfall chart for voice conversations.
 * Displays processing times for ASR, Rasa, and TTS components.
 */
const WaterfallDisplay = ({ latency, sx, ...props }: WaterfallDisplayProps) => {
  const { rasaSpace } = useOurTheme()

  const containerSx = {
    ...sx,
    flexDirection: 'column',
    gap: rasaSpace[1],
  }

  const headerSx = {
    fontSize: 'sm',
    fontWeight: 'bold',
    color: useColorModeValue('gray.700', 'gray.300'),
  }

  const waterfallBarSx = {
    height: '24px',
    borderRadius: '4px',
    overflow: 'hidden',
    border: '1px solid',
    borderColor: useColorModeValue('gray.200', 'gray.600'),
  }

  const legendTableSx = {
    size: 'sm',
    mt: rasaSpace[0.5],
  }

  const getLatencyColor = (type: string) => {
    const colors: { [key: string]: string } = {
      asr: 'blue.500',
      rasa: 'purple.500',
      tts_first: 'orange.500',
      tts_complete: 'green.500',
    }
    return colors[type] || 'gray.500'
  }

  const getLatencyDescription = (type: string) => {
    const descriptions: { [key: string]: string } = {
      asr: 'Time from the first Partial Transcript event to the Final Transcript event from Speech Recognition. It also includes the time taken by the user to speak.',
      rasa: 'Time taken by Rasa to process the text from the Final Transcript event from Speech Recognition until a text response is generated.',
      tts_first:
        'Time between the request sent to Text-to-Speech processing and the first byte of audio received by Rasa.',
      tts_complete:
        'Time taken by Text-to-Speech to complete audio generation. It depends on the length of the text and could overlap with the Bot speaking time.',
    }
    return descriptions[type] || ''
  }

  // Metrics for proportional display (only actual processing latencies)
  const chartMetrics = [
    latency.asr_latency_ms && {
      type: 'asr',
      label: 'ASR',
      value: latency.asr_latency_ms,
    },
    latency.rasa_processing_latency_ms && {
      type: 'rasa',
      label: 'Rasa',
      value: latency.rasa_processing_latency_ms,
    },
    latency.tts_complete_latency_ms && {
      type: 'tts_complete',
      label: 'TTS',
      value: latency.tts_complete_latency_ms,
    },
  ].filter(Boolean)

  // All metrics for legend display
  const allMetrics = [
    latency.asr_latency_ms && {
      type: 'asr',
      label: 'ASR',
      value: latency.asr_latency_ms,
    },
    latency.rasa_processing_latency_ms && {
      type: 'rasa',
      label: 'Rasa',
      value: latency.rasa_processing_latency_ms,
    },
    latency.tts_first_byte_latency_ms && {
      type: 'tts_first',
      label: 'TTS First Byte',
      value: latency.tts_first_byte_latency_ms,
    },
    latency.tts_complete_latency_ms && {
      type: 'tts_complete',
      label: 'TTS Complete',
      value: latency.tts_complete_latency_ms,
    },
  ].filter(Boolean)

  // Calculate total for proportional sizing (only processing latencies)
  const totalLatency = chartMetrics.reduce(
    (sum: number, metric: any) => sum + metric.value,
    0,
  )

  // Calculate total latency for title (Rasa + TTS First Byte)
  const totalDisplayLatency =
    (latency.rasa_processing_latency_ms || 0) +
    (latency.tts_first_byte_latency_ms || 0)

  return (
    <Flex sx={containerSx} {...props}>
      <Text sx={headerSx}>
        Response latency: ~{Math.round(totalDisplayLatency)}ms
      </Text>

      {/* Waterfall Bar */}
      <Box>
        <Flex sx={waterfallBarSx}>
          {chartMetrics.map((metric: any) => {
            const widthPercentage =
              totalLatency > 0 ? (metric.value / totalLatency) * 100 : 0
            return (
              <Tooltip
                key={metric.type}
                label={
                  <Box>
                    <Text fontWeight="bold">
                      {metric.label}: {Math.round(metric.value)}ms
                    </Text>
                    <Text fontSize="xs" mt={1}>
                      {getLatencyDescription(metric.type)}
                    </Text>
                  </Box>
                }
                hasArrow
                placement="top"
              >
                <Box
                  bg={getLatencyColor(metric.type)}
                  width={`${widthPercentage}%`}
                  height="100%"
                  minWidth="40px" // Increased minimum width for better visibility
                  cursor="pointer"
                  _hover={{ opacity: 0.8 }}
                />
              </Tooltip>
            )
          })}
        </Flex>

        {/* Legend */}
        <Table sx={legendTableSx}>
          <Tbody>
            {/* Split metrics into pairs for 2x2 table */}
            {Array.from(
              { length: Math.ceil(allMetrics.length / 2) },
              (_, rowIndex) => (
                <Tr key={rowIndex}>
                  {allMetrics
                    .slice(rowIndex * 2, rowIndex * 2 + 2)
                    .map((metric: any) => (
                      <Td key={metric.type} p={rasaSpace[0.25]} border="none">
                        <Flex align="center" gap={rasaSpace[0.25]}>
                          <Box
                            width="8px"
                            height="8px"
                            bg={getLatencyColor(metric.type)}
                            borderRadius="2px"
                            flexShrink={0}
                          />
                          <Text
                            fontSize="xs"
                            color={useColorModeValue('gray.600', 'gray.400')}
                            noOfLines={1}
                          >
                            {metric.label}: {Math.round(metric.value)}ms
                          </Text>
                        </Flex>
                      </Td>
                    ))}
                  {/* Fill empty cell if odd number of metrics */}
                  {allMetrics.length % 2 !== 0 &&
                    rowIndex === Math.ceil(allMetrics.length / 2) - 1 && (
                      <Td p={rasaSpace[0.25]} border="none" />
                    )}
                </Tr>
              ),
            )}
          </Tbody>
        </Table>
      </Box>
    </Flex>
  )
}

interface LatencyDisplayProps extends FlexProps {
  voiceLatency?: any
  rasaLatency?: any
}

/**
 * Displays processing latency information for the conversation.
 * Shows either a detailed waterfall chart for voice conversations or
 * a simpler display for text-only conversations.
 */
export const LatencyDisplay = ({
  sx,
  voiceLatency,
  rasaLatency,
  ...props
}: LatencyDisplayProps) => {
  // We give priority to voice latency as most comprehensive.
  if (isVoiceLatency(voiceLatency)) {
    return <WaterfallDisplay latency={voiceLatency} sx={sx} {...props} />
  }

  // If voice latency is not available, we use rasa latency.
  if (isRasaLatency(rasaLatency)) {
    return <MinimalDisplay latency={rasaLatency} sx={sx} {...props} />
  }

  // If rasa latency is not available, we are either waiting for first data from voice stream or the conversation is not started yet.
  if (!rasaLatency) {
    return <Text>Rasa latency data is not available yet</Text>
  }

  // If rasa latency is available but did not pass type guard, we are in an unexpected state
  // and we should log an error but gracefully fallback to showing no data.
  console.warn(
    `Latency data has unexpected format. Raw data of rasaLatency: ${rasaLatency},
    raw data of voice latency: ${voiceLatency}`,
  )

  return <Text>Latency data is not available yet</Text>
}
