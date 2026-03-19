import { forwardRef } from "react";
import { type BoxProps, type FlexProps, Box, Flex } from "@chakra-ui/react";
import { UtteranceType, type Utterance } from "../../../types";
import { Avatar } from "../../../Avatar";
import { UserAvatar, BotAvatar } from "../../../assets/images";
import { useConversationLogSx } from "../useConversationLogSx";
import { isBotUtteranceEmpty } from "../../../utils";

interface Props extends FlexProps {
  utterance?: Omit<Utterance, "entities">;
  previousUtterance?: Omit<Utterance, "entities">;
  isSelected?: boolean;
  isHighlighted?: boolean;
  topIntentName?: string;
  containerSx?: FlexProps;
  messageSx?: BoxProps;
  messageBreakout?: React.ReactNode;
  isInteractive?: boolean;
  isUser?: boolean;
  topOverlay?: React.ReactNode;
}

export const MessageMarkup = forwardRef<HTMLDivElement | null, Props>(
  (props: Props, ref) => {
    const {
      utterance,
      isSelected,
      isHighlighted,
      previousUtterance,
      topIntentName,
      children,
      containerSx: containerSxAdditional,
      messageSx: messageSxAdditional,
      messageBreakout,
      isInteractive = false,
      isUser = false,
      topOverlay,
      ...otherProps
    } = props;

    const { hoverableSx } = useConversationLogSx(isSelected ?? false);

    const regularBgColor = "rasaNeutral.100";
    const userBgColor = "rasawebDeepPurple.800";
    const regularTextColor = "rasawebDeepPurple.800";
    const highlightTextColor = "rasaNeutral.50";
    const selectedBgColor = "rasaNeutral.50";

    const utteranceType = utterance?.type || UtteranceType.Bot;

    const showAvatar =
      !(utterance && isBotUtteranceEmpty(utterance)) &&
      utteranceType !== previousUtterance?.type;

    const containerSx = {
      _first: { mt: 0 },
      px: "1rem",
      py: "0.5rem",
      pr: isUser ? "1.5rem" : "3.5rem",
      pl: isUser ? "3.5rem" : "1.5rem",
      bg: isSelected ? "rasaNeutral.200" : "transparent",
      ...hoverableSx,
      ...containerSxAdditional,
    };

    let messageSxBg;

    if (isUser) {
      messageSxBg = userBgColor;
    } else {
      messageSxBg = isSelected ? selectedBgColor : regularBgColor;
    }

    const messageSxColor =
      isUser || (!isInteractive && isHighlighted)
        ? highlightTextColor
        : regularTextColor;

    const messageSx = {
      borderRadius: isUser
        ? "1rem 0.25rem 1rem 1rem"
        : "0.25rem 1rem 1rem 1rem",
      p: utterance ? "0" : "1rem",
      bg: messageSxBg,
      color: messageSxColor,
      fontSize: "0.813rem",
      overflow: "hidden",
      mb: "0.25rem",
      wordBreak: "break-word",
      maxWidth: "20rem",
      ...messageSxAdditional,
    };

    const intentSx = {
      color: "#6B7694",
      fontSize: "0.75rem",
      mb: "0.5rem",
      mr: "1rem",
      _last: { mb: 0 },
    };

    const getDataTestId = () => {
      if (!utterance) {
        return "agent-loading-spinner";
      }
      if (isHighlighted) {
        return "conversation-message-highlighted";
      }
      if (isUser) {
        return "assistant-user-message";
      }
      return "assistant-response";
    };

    return (
      <Flex css={containerSx} {...otherProps} ref={ref} position="relative">
        {showAvatar ? (
          <Avatar
            data-testid="message-avatar"
            size="sm"
            mr={isUser ? 0 : "0.5rem"}
            ml={isUser ? "0.5rem" : 0}
            src={isUser ? UserAvatar : BotAvatar}
          />
        ) : (
          <Box width="2.5rem" />
        )}
        <Box display="inline-block">
          {topOverlay}
          <Box
            css={messageSx}
            data-testid={getDataTestId()}
            className={isUser ? "" : "message-bubble"} // styling hack, see useConversationLogSx.tsx > hoverableSx
          >
            {children}
          </Box>
          {messageBreakout}
          {!!topIntentName && <Box css={intentSx}>{topIntentName}</Box>}
        </Box>
      </Flex>
    );
  },
);
