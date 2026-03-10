import { Box, Button } from "@chakra-ui/react";
import { Icon } from "../Icon";
import type { ConversationEventAction, UnionEventType } from "../types";

type ConversationEventActionButtonProps = {
  actions?: ConversationEventAction[];
  event: UnionEventType;
};

export const ConversationEventActionButton = ({ actions, event }: ConversationEventActionButtonProps) => {
  if (!actions || actions.length === 0) {
    return null;
  }

  return (
    <Box
      position="absolute"
      top="-1.25rem"
      right="5"
      zIndex={1}
      bg="rasaNeutral.50"
      boxShadow="0px 2px 8px 0px #00000026"
      borderRadius="lg"
    >
      {actions.map(({ icon, label, action }, index: number) => (
        /* TODO: style when more than one actions */
        <Button
          key={index}
          variant="subtle"
          colorPalette="dark"
          size="sm"
          onClick={() => action({ ...event })}
          aria-label={label}
        >
          <Icon icon={icon} />
          {label}
        </Button>
      ))}
    </Box>
  );
};
