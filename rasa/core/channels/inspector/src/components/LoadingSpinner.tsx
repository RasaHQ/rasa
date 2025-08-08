import {
  Center,
  Spinner,
  Text,
  Button,
  useColorModeValue,
} from '@chakra-ui/react'
import { useOurTheme } from '../theme'
import { createAudioConnection } from '../helpers/audio/audiostream.ts'

interface LoadingSpinnerProps {
  onLatencyUpdate?: (latency: any) => void
}

export const LoadingSpinner = ({ onLatencyUpdate }: LoadingSpinnerProps) => {
  const { rasaSpace } = useOurTheme()
  const isVoice = window.location.href.includes('browser_audio')
  const text = isVoice
    ? 'Start a new conversation'
    : 'Waiting for a new conversation'
  return (
    <Center height={'100vh'} flexDirection="column">
      <Spinner
        speed="1s"
        emptyColor={useColorModeValue('neutral.500', 'neutral.500')}
        color={useColorModeValue('rasaPurple.800', 'rasaPurple.800')}
        size="lg"
        mb={rasaSpace[1]}
      />
      <Text fontSize="lg">{text}</Text>
      {isVoice ? (
        <Button
          onClick={async () =>
            await createAudioConnection(window.location.href, onLatencyUpdate)
          }
        >
          Go
        </Button>
      ) : null}
    </Center>
  )
}
