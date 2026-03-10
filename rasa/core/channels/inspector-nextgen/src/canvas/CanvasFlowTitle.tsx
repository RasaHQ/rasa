import { Heading, HStack, Text } from "@chakra-ui/react";

export const CanvasFlowTitle = ({ flowName }: { flowName: string }) => {
  const containerSx = {
    position: "absolute",
    top: "1rem",
    left: "1.5rem",
    height: "auto",
    flexDirection: "row",
    maxWidth: "calc(100% - 3rem)",
  };

  return (
    <HStack css={containerSx} fontSize="0.813rem">
      <Heading size="md">Current flow: </Heading>
      <Text
        size="md"
        overflow={"hidden"}
        textOverflow={"ellipsis"}
        whiteSpace={"nowrap"}
      >
        {flowName}
      </Text>
    </HStack>
  );
};
