import { Box, Flex } from "@chakra-ui/react";
import { MessageMarkup } from "./ConversationLog/Message/MessageMarkup";

export const ConversationLoadingSpinner = () => {
  const loadingDotSx = {
    borderRadius: "50%",
    width: "0.375rem",
    height: "0.375rem",
    bg: "rasaNeutral.500",
    animationName: "blink",
    animationDuration: "1.25s",
    animationTimingFunction: "ease-in-out",
    animationIterationCount: "infinite",
  };

  return (
    <MessageMarkup aria-label="Agent is typing">
      <Flex gap="0.25rem" data-testid="loading-dots">
        <Box css={loadingDotSx} />
        <Box css={loadingDotSx} animationDelay="0.25s" />
        <Box css={loadingDotSx} animationDelay="0.5s" />
      </Flex>
    </MessageMarkup>
  );
};
