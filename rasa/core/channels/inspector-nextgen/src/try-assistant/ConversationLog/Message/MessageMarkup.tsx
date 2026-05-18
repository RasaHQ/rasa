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
  inspectorMode?: boolean;
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
      inspectorMode,
      isInteractive = false,
      isUser = false,
      topOverlay,
      ...otherProps
    } = props;

    const shouldUseSelectedStyles = isSelected && inspectorMode;

    const { hoverableSx } = useConversationLogSx(shouldUseSelectedStyles ?? false);

    const regularBgColor = "bg.muted";
    const userBgColor = "fg";
    const regularTextColor = "fg";
    const highlightTextColor = "bg.subtle";
    const selectedBgColor = "bg.panel";

    const utteranceType = utterance?.type || UtteranceType.Bot;

    const showAvatar =
      !(utterance && isBotUtteranceEmpty(utterance)) &&
      utteranceType !== previousUtterance?.type;

    const containerSx = {
      _first: { mt: 0 },
      px: "4",
      py: "2",
      pr: isUser ? "6" : "14",
      pl: isUser ? "14" : "6",
      bg: shouldUseSelectedStyles ? "bg.muted" : "transparent",
      ...(inspectorMode ? hoverableSx : {}),
      ...containerSxAdditional,
    };

    let messageSxBg;

    if (isUser) {
      messageSxBg = userBgColor;
    } else {
      messageSxBg = shouldUseSelectedStyles ? selectedBgColor : regularBgColor;
    }

    const messageSxColor =
      isUser || (!isInteractive && isHighlighted)
        ? highlightTextColor
        : regularTextColor;

    const messageSx = {
      borderTopLeftRadius: isUser ? "2xl" : "sm",
      borderTopRightRadius: isUser ? "sm" : "2xl",
      borderBottomRightRadius: "2xl",
      borderBottomLeftRadius: "2xl",
      p: utterance ? "0" : "4",
      bg: messageSxBg,
      color: messageSxColor,
      overflow: "hidden",
      mb: "1",
      wordBreak: "break-word",
      maxWidth: "80",
      ...messageSxAdditional,
    };

    const intentSx = {
      color: "fg.muted",
      mb: "2",
      mr: "4",
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
          <Box width="10" />
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
