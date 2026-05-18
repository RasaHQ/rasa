import { Flex, Text } from "@chakra-ui/react";
import { useInspectorStore } from "../store";
import { ViewControl } from "./ViewControl";

export const StandaloneHeader = () => {
  const assistantId = useInspectorStore((s) => s.assistantId);

  const styles = {
    height: "12",
    boxShadow: "header",
    borderBottom: "1px solid",
    borderColor: "border.emphasized",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    flexShrink: "0",
    px: "6",
    bg: "bg.panel",
  };

  return (
    <Flex css={styles}>
      {assistantId && (
        <Text textStyle="sm" position="absolute" left="6">
          {assistantId}
        </Text>
      )}

      <ViewControl />
    </Flex>
  );
};
