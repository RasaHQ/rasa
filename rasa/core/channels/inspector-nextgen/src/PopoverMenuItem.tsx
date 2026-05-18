import { Flex, Menu, Text } from "@chakra-ui/react";
import { Icon } from "./Icon";

export interface PopoverMenuItemProps {
  icon: Parameters<typeof Icon>[0]["icon"];
  label: string;
  onClick: () => void;
  testId?: string;
  isSelected?: boolean;
}

export const PopoverMenuItem = ({
  icon,
  label,
  onClick,
  testId,
}: PopoverMenuItemProps) => {
  return (
    <Menu.Item
      onClick={onClick}
      data-testid={
        testId ?? `popover-menu-${label.toLowerCase().replace(/\s+/g, "-")}`
      }
      value={label}
    >
      <Flex
        width="4"
        height="4"
        display="flex"
        alignItems="center"
        justifyContent="center"
      >
        <Icon icon={icon} />
      </Flex>
      <Text textStyle="sm">
        {label}
      </Text>
    </Menu.Item>
  );
};
