// RecruitmentPanel.tsx
import { Box, Flex, Heading, IconButton, Text, Button } from '@chakra-ui/react'
import { CloseIcon } from '@chakra-ui/icons'
import { useOurTheme } from '../theme'

interface RecruitmentPanelProps {
  onClose: () => void
}

export const RecruitmentPanel: React.FC<RecruitmentPanelProps> = ({
  onClose,
}) => {
  const { rasaRadii } = useOurTheme()

  const textColor = 'white'
  const iconColor = 'white'

  const boxSx = {
    borderRadius: rasaRadii.normal,
    padding: '1rem',
    bgGradient: 'linear(to-b, #4E61E1, #7622D2)',
    color: textColor,
  }

  return (
    <Box sx={boxSx}>
      <Flex justify="space-between" align="center">
        <Flex align="center">
          <Heading as="h3" size="lg" fontWeight="bold" color={textColor}>
            Help us Improve Rasa Pro!
          </Heading>
        </Flex>
        <IconButton
          aria-label="Close"
          icon={<CloseIcon color={iconColor} />}
          size="sm"
          onClick={onClose}
          bg="transparent"
          _hover={{ bg: 'rgba(255, 255, 255, 0.2)' }}
        />
      </Flex>
      <Flex align="center" mt="0.5rem" justify="space-between">
        <Text fontSize="sm" color={textColor}>
          We're looking for users to share their feedback.
        </Text>
        <Button
          as="a"
          href="https://feedback.rasa.com"
          target="_blank"
          rel="noopener noreferrer"
          color="#7622D2"
          bg="white"
          fontWeight="bold"
          ml="1rem"
          size="sm"
          _hover={{ bg: 'whiteAlpha.800' }}
        >
          Sign up
        </Button>
      </Flex>
    </Box>
  )
}
