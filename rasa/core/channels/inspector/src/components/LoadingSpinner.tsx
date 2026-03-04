import { useEffect, useState } from 'react'
import {
  Center,
  Spinner,
  Text,
  Button,
  Select,
  useColorModeValue,
} from '@chakra-ui/react'
import { useOurTheme } from '../theme'
import {
  createAudioConnection,
  fetchSupportedLanguages,
} from '../helpers/audio/audiostream.ts'

interface LoadingSpinnerProps {
  onLatencyUpdate?: (latency: any) => void
}

export const LoadingSpinner = ({ onLatencyUpdate }: LoadingSpinnerProps) => {
  const { rasaSpace } = useOurTheme()
  const isVoice = window.location.href.includes('browser_audio')
  const [languages, setLanguages] = useState<string[]>([])
  const [selectedLanguage, setSelectedLanguage] = useState<string>('')

  useEffect(() => {
    if (!isVoice) return
    fetchSupportedLanguages(window.location.origin)
      .then((langs) => {
        setLanguages(langs)
        setSelectedLanguage((prev) => (prev && langs.includes(prev) ? prev : langs[0] ?? ''))
      })
      .catch(() => setLanguages([]))
  }, [isVoice])

  return (
    <Center height={'100vh'} flexDirection="column">
      {!isVoice ? (
        <>
          <Spinner
            speed="1s"
            emptyColor={useColorModeValue('neutral.500', 'neutral.500')}
            color={useColorModeValue('rasaPurple.800', 'rasaPurple.800')}
            size="lg"
            mb={rasaSpace[1]}
          />
          <Text fontSize="lg">Waiting for a new conversation</Text>
        </>
      ) : (
        <>
          <Text fontSize="lg">Start a new conversation.</Text>
          {languages.length > 1 && (
            <Select
              mt={rasaSpace[1]}
              maxW="xs"
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              placeholder="Select language"
            >
              {languages.map((lang) => (
                <option key={lang} value={lang}>
                  {lang}
                </option>
              ))}
            </Select>
          )}
          <Button
            onClick={async () =>
              await createAudioConnection(
                window.location.origin,
                onLatencyUpdate,
                selectedLanguage || undefined,
              )
            }
            mt={rasaSpace[1]}
          >
            Go
          </Button>
        </>
      )}
    </Center>
  )
}
