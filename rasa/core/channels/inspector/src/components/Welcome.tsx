import { Box, Flex, FlexProps, Heading, Link, Text } from '@chakra-ui/react'
import { RasaLogo, RasaLogoDark } from './RasaLogo'
import { useOurTheme } from '../theme'

interface WelcomeProps extends FlexProps {
  isRecruitmentVisible?: boolean
}

export const Welcome = ({
  sx,
  isRecruitmentVisible,
  ...props
}: WelcomeProps) => {
  const { rasaSpace } = useOurTheme()

  const containerSx = {
    ...sx,
    color: isRecruitmentVisible ? 'black' : 'neutral.50',
    bg: isRecruitmentVisible ? 'white' : undefined,
    bgGradient: isRecruitmentVisible
      ? undefined
      : 'linear(to-b, #4E61E1, #7622D2)',
  }

  const linkSx = {
    flexGrow: 0,
    color: isRecruitmentVisible ? '#0000EE' : 'neutral.50',
    textDecoration: 'underline',
    _hover: {
      color: isRecruitmentVisible ? 'link.visited' : 'neutral.400',
    },
  }

  return (
    <Flex sx={containerSx} {...props}>
      <Box>
        <Heading as="h1" size="xl" mb={rasaSpace[1]}>
          Rasa Inspector
        </Heading>
        <Text as="span">New to the Inspector?</Text>
        <Link
          sx={linkSx}
          href="https://rasa.com/docs/rasa-pro/production/inspect-assistant/"
          target="_blank"
          ml={rasaSpace[0.25]}
        >
          Browse the docs
        </Link>
      </Box>
      {isRecruitmentVisible ? (
        <RasaLogoDark sx={{ flexShrink: 0, marginLeft: 'auto' }} />
      ) : (
        <RasaLogo sx={{ flexShrink: 0, marginLeft: 'auto' }} />
      )}
    </Flex>
  )
}
