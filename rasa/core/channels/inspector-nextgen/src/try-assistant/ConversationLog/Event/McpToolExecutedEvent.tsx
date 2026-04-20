import { Box, Flex, Text } from "@chakra-ui/react";
import React, { forwardRef } from "react";
import type {
  ConversationEvent,
  ConversationEventAction,
} from "../../../types";
import { Icon } from "../../../Icon";
import { Wrench } from "../../../Icon/icons";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";
import { useConversationLogSx } from "../useConversationLogSx";
import { ErrorDot } from "./ErrorDot";

interface McpToolExecutedEventProps {
  event: ConversationEvent;
  isSelected: boolean;
  conversationEventActions?: ConversationEventAction[];
  onClick?: React.MouseEventHandler<HTMLDivElement>;
  onKeyDown?: (e: React.KeyboardEvent<HTMLDivElement>) => void;
}

export const McpToolExecutedEvent = forwardRef<
  HTMLDivElement | null,
  McpToolExecutedEventProps
>((props: McpToolExecutedEventProps, ref) => {
  const {
    event,
    isSelected = false,
    conversationEventActions,
    onClick,
    onKeyDown,
    ...otherProps
  } = props;
  const { hoverableContainerSx, baseMessageSx, iconSx } =
    useConversationLogSx(isSelected);
  const [isHovered, setIsHovered] = React.useState(false);

  const toolName = event.metadata?.tool_name ?? event.name ?? "tool";
  const isError = event.metadata?.tool_is_error === true;

  return (
    <Flex
      ref={ref}
      tabIndex={0}
      role="button"
      position="relative"
      cursor="pointer"
      css={hoverableContainerSx}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
      onKeyDown={onKeyDown}
      {...otherProps}
    >
      {isHovered && (
        <ConversationEventActionButton
          event={event}
          actions={conversationEventActions}
        />
      )}
      <Box css={baseMessageSx}>
        <Icon icon={Wrench} style={iconSx} />
        <Text size="sm" lineClamp={2} wordBreak="break-all" variant="muted">
          Tool{" "}
          <Text as="span" fontWeight="500" variant="muted">
            {toolName}
          </Text>{" "}
          used
        </Text>
        {isError && <ErrorDot />}
      </Box>
    </Flex>
  );
});

McpToolExecutedEvent.displayName = "McpToolExecutedEvent";
