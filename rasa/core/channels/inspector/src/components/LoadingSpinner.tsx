import { Center, Spinner, Text, useColorModeValue } from "@chakra-ui/react";
import { useOurTheme } from "../theme";

export const LoadingSpinner = () => {
  const { rasaSpace } = useOurTheme();

  return (
    <Center height={"100vh"} flexDirection="column">
      <Spinner
        speed="1s"
        emptyColor={useColorModeValue("neutral.500", "neutral.500")}
        color={useColorModeValue("rasaPurple.800", "rasaPurple.800")}
        size="lg"
        mb={rasaSpace[1]}
      />
      <Text>Loading</Text>
    </Center>
  );
};
