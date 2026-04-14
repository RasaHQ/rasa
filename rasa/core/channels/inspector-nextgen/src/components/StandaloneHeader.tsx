import { Flex, Text } from "@chakra-ui/react";
import { useInspectorStore } from "../store";
import { ViewControl } from "./ViewControl";

export const StandaloneHeader = () => {
  const assistantId = useInspectorStore((s) => s.assistantId);

  const styles = {
    height: "3rem",
    bg: "rasaNeutral.50",
    boxShadow: "header",
    borderBottom: "1px solid",
    borderColor: "rasaNeutral.400",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    flexShrink: "0",
    px: "1.5rem",
  };

  return (
    <Flex css={styles}>
      {assistantId && (
        <Text size="sm" position="absolute" left="1.5rem">
          {assistantId}
        </Text>
      )}

      <ViewControl />
    </Flex>
  );
};
