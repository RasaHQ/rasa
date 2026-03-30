import { Box, Text } from "@chakra-ui/react";
import { useState } from "react";
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
  isSelected,
}: PopoverMenuItemProps) => {
  const [isHovered, setIsHovered] = useState(false);
  return (
    <Box
      as="button"
      display="flex"
      alignItems="center"
      gap="0.75rem"
      width="100%"
      px="0.75rem"
      py="0.5rem"
      cursor="pointer"
      bg={isSelected ? "rasaNeutral.100" : "transparent"}
      _hover={{ bg: "rasaNeutral.100" }}
      onClick={onClick}
      data-testid={testId ?? `popover-menu-${label.toLowerCase().replace(/\s+/g, "-")}`}
      fontSize="1rem"
      borderRadius="0.5rem"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Box width="1rem" height="1rem" display="flex" alignItems="center" justifyContent="center">
        <Icon color={isHovered ? "" : "rasaNeutral.700"} icon={icon} />
      </Box>
      <Text size="sm" color={isHovered ? "rasawebDeepPurple.900" : "rasawebDeepPurple.800"}>
        {label}
      </Text>
    </Box>
  );
};
