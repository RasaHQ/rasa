import { Button, type ButtonProps } from "@chakra-ui/react";
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
  const sx = {
    borderColor: "rasaNeutral.300",
    color: "rasawebDeepPurple.800",
    _hover: {
      bg: "rasaNeutral.50",
    },
  };
  const activeSx = {
    color: "rasaNeutral.50",
    bg: "rasawebPurple.800",
    borderColor: "rasawebPurple.800",
    _hover: {
      bg: "rasawebPurple.800",
      color: "rasaNeutral.50",
    },
  };
  return (
    <Button
      {...props}
      rounded="full"
      variant="outline"
      size="sm"
      height="1.5rem"
      fontSize="13px"
      px={2.5}
      css={isActive ? { ...sx, ...activeSx } : sx}
      onClick={() => onActivate(item.value)}
    >
      <Icon icon={LocationCrosshairs} />
      {item.label}
    </Button>
  );
};

