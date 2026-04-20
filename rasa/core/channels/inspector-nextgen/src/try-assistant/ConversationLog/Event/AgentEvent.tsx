import { Box, Flex, Text } from "@chakra-ui/react";
import React, { forwardRef } from "react";
import {
  ConversationEventType,
  type ConversationEvent,
  type ConversationEventAction,
} from "../../../types";
import { Icon } from "../../../Icon";
import { Robot } from "../../../Icon/icons";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";
import { useConversationLogSx } from "../useConversationLogSx";
import { ErrorDot } from "./ErrorDot";

const agentEventText: Partial<Record<ConversationEventType, string>> = {
  [ConversationEventType.AgentStarted]: "invoked",
  [ConversationEventType.AgentResumed]: "resumed",
  [ConversationEventType.AgentCompleted]: "stopped",
  [ConversationEventType.AgentCancelled]: "stopped",
  [ConversationEventType.AgentInterrupted]: "interrupted",
};

interface AgentEventProps {
  event: ConversationEvent;
  isSelected: boolean;
  conversationEventActions?: ConversationEventAction[];
  onClick?: React.MouseEventHandler<HTMLDivElement>;
  onKeyDown?: (e: React.KeyboardEvent<HTMLDivElement>) => void;
}

export const AgentEvent = forwardRef<HTMLDivElement | null, AgentEventProps>(
  (props: AgentEventProps, ref) => {
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

    const agentId = event.metadata?.agent_id ?? event.name ?? "agent";
    const statusText = agentEventText[event.conversationEventType] ?? "invoked";
    const isError = event.metadata?.execution_success === false;

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
          <Icon icon={Robot} style={iconSx} />
          <Text size="sm" lineClamp={2} wordBreak="break-all" variant="muted">
            Sub-agent{" "}
            <Text as="span" fontWeight="500" variant="muted">
              {agentId}
            </Text>{" "}
            {statusText}
          </Text>
          {isError && <ErrorDot />}
        </Box>
      </Flex>
    );
  },
);

AgentEvent.displayName = "AgentEvent";
