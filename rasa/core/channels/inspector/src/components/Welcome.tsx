import {
  Box,
  Flex,
  FlexProps,
  Heading,
  Link,
  Text,
  useColorModeValue,
} from "@chakra-ui/react";
import { RasaLogo } from "./RasaLogo";
import { useOurTheme } from "../theme";

export const Welcome = ({ sx, ...props }: FlexProps) => {
  const { rasaSpace } = useOurTheme();

  const containerSx = {
    ...sx,
    color: useColorModeValue("neutral.50", "neutral.50"),
    bgGradient: "linear(to-b, #4E61E1, #7622D2)",
  };

  const linkSx = {
    flexGrow: 0,
    color: useColorModeValue("neutral.50", "neutral.50"),
    _hover: {
      color: useColorModeValue("neutral.400", "neutral.400"),
    },
  };

  const logoSx = {
    flexShrink:0,
    marginLeft: "auto"
  };

  return (
    <Flex sx={containerSx} {...props}>
      <Box>
        <Heading as="h1" size="xl" mb={rasaSpace[1]}>
          Rasa Inspector
        </Heading>
        <Text as="span">
          New to the Inspector?
        </Text>
        <Link
          sx={linkSx}
          href="https://rasa.com/docs/rasa-pro/production/inspect-assistant/"
          target="_blank"
          ml={rasaSpace[0.25]}
        >Browse the docs</Link>
      </Box>
      <RasaLogo sx={logoSx}/>
    </Flex>
  );
};
