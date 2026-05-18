import { Box, Flex } from "@chakra-ui/react";
import { MessageMarkup } from "./ConversationLog/Message/MessageMarkup";

export const ConversationLoadingSpinner = () => {
  const loadingDotSx = {
    borderRadius: "full",
    width: "1.5",
    height: "1.5",
    bg: "border.emphasized",
    animationName: "blink",
    animationDuration: "1.25s",
    animationTimingFunction: "ease-in-out",
    animationIterationCount: "infinite",
  };

  return (
    <MessageMarkup aria-label="Agent is typing">
      <Flex gap="1" data-testid="loading-dots">
        <Box css={loadingDotSx} />
        <Box css={loadingDotSx} animationDelay="0.25s" />
        <Box css={loadingDotSx} animationDelay="0.5s" />
      </Flex>
    </MessageMarkup>
  );
};
