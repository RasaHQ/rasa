import { Flex, Text } from "@chakra-ui/react";
import { useInspectorStore } from "../store";
import { ViewControl } from "./ViewControl";

export const StandaloneHeader = () => {
  const assistantId = useInspectorStore((s) => s.assistantId);

  return (
    <Flex
      height="3rem"
      flexShrink={0}
      bg="rasaNeutral.50"
      boxShadow="header"
      borderBottom="1px solid"
      borderColor="rasaNeutral.400"
      alignItems="center"
      justifyContent="center"
      position="relative"
      px="1.5rem"
    >
      {assistantId && (
        <Text position="absolute" left="1.5rem">
          {assistantId}
        </Text>
      )}

      <ViewControl />
    </Flex>
  );
};
