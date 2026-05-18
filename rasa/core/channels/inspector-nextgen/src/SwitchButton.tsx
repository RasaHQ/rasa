import { Button, Toggle, type ButtonProps } from "@chakra-ui/react";
import { Icon, LocationCrosshairs } from "./Icon";

type Props = ButtonProps & {
  item: {
    value: string;
    label: string;
  };
  onActivate: (value: string) => void;
  isActive?: boolean;
};

export const SwitchButton = ({
  item,
  isActive,
  onActivate,
  ...props
}: Props) => {
  return (
    <Toggle.Root
      pressed={isActive}
      onPressedChange={() => onActivate(item.value)}
      asChild
    >
      <Button
        {...props}
        colorPalette={{ base: "gray", _pressed: "purple" }}
        variant={{ base: "outline", _pressed: "solid" }}
        rounded="full"
        size="sm"
        height="6"
        px={2.5}
      >
        <Icon icon={LocationCrosshairs} />
        {item.label}
      </Button>
    </Toggle.Root>
  );
};
