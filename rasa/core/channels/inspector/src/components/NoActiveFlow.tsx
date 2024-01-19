import { Center, CenterProps, Text, useColorModeValue } from "@chakra-ui/react";
import { useOurTheme } from "../theme";
import { SaraDiagrams } from "./SaraDiagrams";

export const NoActiveFlow = (props: CenterProps) => {
  const { rasaSpace, rasaFontSizes } = useOurTheme();
  const textColor = useColorModeValue("neutral.700", "neutral.700");

  return (
    <Center height={"100%"} flexDirection="column" {...props}>
      <SaraDiagrams mb={rasaSpace[1]} size={220} />
      <Text as={"b"} fontSize={rasaFontSizes.lg} mb={rasaSpace[0.5]}>
        No flow is currently active
      </Text>
      <Text maxW="21.25rem" color={textColor}>
        Type a message to your assistant in the chat window on the right to
        activate one
      </Text>
    </Center>
  );
};
