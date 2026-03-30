import { Box, Flex, IconButton, Text } from "@chakra-ui/react";
import React, { forwardRef } from "react";
import { Icon } from "../../../Icon";
import {
  CircleExclamation,
  ClockRotateLeft,
  Code,
  Ear,
} from "../../../Icon/icons";
import { type ConversationEvent, type ConversationEventAction } from "../../../types";
import { isInternalRasaAction } from "../../../utils";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";
import { useConversationLogSx } from "../useConversationLogSx";

const ACTION_LISTEN_NAME = "action_listen";

interface ActionEventProps {
  event: ConversationEvent;
  isSelected: boolean;
  replayConversation?: (eventId: string) => void;
  conversationEventActions?: ConversationEventAction[];
}

export const ActionEvent = forwardRef<HTMLDivElement | null, ActionEventProps>(
  (props: ActionEventProps, ref) => {
    const {
      event,
      isSelected = false,
      replayConversation,
      conversationEventActions,
      ...otherProps
    } = props;
    const { hoverableContainerSx, containerSx, baseMessageSx, iconSx } =
      useConversationLogSx(isSelected);
    const [isHovered, setIsHovered] = React.useState(false);
    const shouldShowReplayButton = false;
    //event.name === ACTION_LISTEN_NAME && replayConversation;

    // checking for false explicitly because metadata is not always present
    const shouldShowActionError = event.metadata?.execution_success === false;

    const nonClickableContainerSx = {
      ...containerSx,
      // makes sure utterances are displayed closer to a bot message they trigger
      pb: 0,
    };

    const clickableContainerSx = {
      ...hoverableContainerSx,
      ":hover .replayIcon": {
        display: "inline-block",
      },
      // the replay button is a bit too big for the container, so we reduce the padding
      // to get the same row height as the other events
      pt: shouldShowReplayButton ? "1px" : undefined,
      pb: shouldShowReplayButton ? "1px" : undefined,
    };

    const errorIconSx = {
      color: "#A72E2C",
      marginLeft: "0.25rem",
    };

    const actionNameSx = {
      ...baseMessageSx,
      color: "rasaNeutral.700",
    };

    const utterSx = {
      ...actionNameSx,
      ml: "2.75rem",
    };

    const replaySx = {
      display: "inline-block",
      ml: "auto",
    };

    const actionIcon = event.name === ACTION_LISTEN_NAME ? Ear : Code;
    const actionLabel =
      event.name === ACTION_LISTEN_NAME ? "Waiting for user input" : event.name;

    const handleReplayClick = (
      mouseEvent: React.MouseEvent<HTMLButtonElement>,
    ) => {
      if (replayConversation) {
        replayConversation(event.id);
      }
      mouseEvent.stopPropagation();
    };

    if (isInternalRasaAction(event.name || "")) {
      return null;
    } else if (event.name?.startsWith("utter_")) {
      return (
        <Flex css={nonClickableContainerSx} ref={ref} {...otherProps}>
          <Text size="sm" css={utterSx}>
            {event.name}
          </Text>
        </Flex>
      );
    } else {
      return (
        <Flex
          css={clickableContainerSx}
          alignItems="center"
          ref={ref}
          tabIndex={0}
          position="relative"
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          {...otherProps}
        >
          {isHovered && <ConversationEventActionButton event={event} actions={conversationEventActions} />}
          <Box css={actionNameSx}>
            <Icon icon={actionIcon} style={iconSx} />
            <Text
              size="sm"
              variant="muted"
              lineClamp={2}
              wordBreak={"break-all"}
            >
              {actionLabel}
            </Text>
          </Box>
          {shouldShowActionError ? (
            <Icon
              icon={CircleExclamation}
              style={errorIconSx}
              box-size="1rem"
              aria-label="Action failed"
            />
          ) : null}
          {shouldShowReplayButton ? (
            <IconButton
              variant="ghost"
              colorScheme="neutral"
              size="sm"
              aria-label="Replay conversation"
              css={replaySx}
              onClick={handleReplayClick}
            >
              Replay
              <Icon icon={ClockRotateLeft} />
            </IconButton>
          ) : null}
        </Flex>
      );
    }
  },
);
