import { Box, Heading, Popover } from "@chakra-ui/react";
import { Tooltip } from "./Tooltip";

interface PopoverMenuProps {
  trigger: React.ReactNode;
  header: string;
  children: React.ReactNode;
  placement?: "bottom-end" | "bottom-start" | "bottom" | "top" | "top-end" | "top-start";
  tooltipContent?: string;
}

export const PopoverMenu = ({
  trigger,
  header,
  children,
  placement = "bottom-end",
  tooltipContent,
}: PopoverMenuProps) => {
  const triggerElement = (
    <Popover.Trigger asChild>
      {trigger}
    </Popover.Trigger>
  );

  return (
    <Popover.Root positioning={{ placement }}>
      {tooltipContent ? (
        <Tooltip content={tooltipContent} showArrow>
          <Box>{triggerElement}</Box>
        </Tooltip>
      ) : (
        triggerElement
      )}
      <Popover.Positioner>
        <Popover.Content
          width="240px"
          borderRadius="0.5rem"
          boxShadow="0px 2px 10px 0px rgba(0, 0, 0, 0.18)"
        >
          <Popover.Body p="0.5rem">
            <Box px="0.75rem" py="0.5rem" textAlign="start">
              <Heading size="sm">{header}</Heading>
            </Box>
            {children}
          </Popover.Body>
        </Popover.Content>
      </Popover.Positioner>
    </Popover.Root>
  );
};
