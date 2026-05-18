import { Box, Menu } from "@chakra-ui/react";
import { Tooltip } from "./Tooltip";

interface PopoverMenuProps {
  trigger: React.ReactNode;
  header: string;
  children: React.ReactNode;
  placement?:
  | "bottom-end"
  | "bottom-start"
  | "bottom"
  | "top"
  | "top-end"
  | "top-start";
  tooltipContent?: string;
}

export const PopoverMenu = ({
  trigger,
  header,
  children,
  placement = "bottom-end",
  tooltipContent,
}: PopoverMenuProps) => {
  const triggerElement = <Menu.Trigger asChild>{trigger}</Menu.Trigger>;

  return (
    <Menu.Root positioning={{ placement }}>
      {tooltipContent ? (
        <Tooltip content={tooltipContent} showArrow>
          <Box>{triggerElement}</Box>
        </Tooltip>
      ) : (
        triggerElement
      )}
      <Menu.Positioner>
        <Menu.Content width="56">
          <Menu.ItemGroup>
            <Menu.ItemGroupLabel>{header}</Menu.ItemGroupLabel>
            {children}
          </Menu.ItemGroup>
        </Menu.Content>
      </Menu.Positioner>
    </Menu.Root>
  );
};
