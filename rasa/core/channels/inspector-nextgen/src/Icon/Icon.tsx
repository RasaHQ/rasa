import {
  FontAwesomeIcon,
  type FontAwesomeIconProps,
} from "@fortawesome/react-fontawesome";
import type { IconDefinition } from "@fortawesome/fontawesome-svg-core";
import { useToken } from "@chakra-ui/react";

interface Props extends Omit<FontAwesomeIconProps, "icon" | "color"> {
  readonly icon: IconDefinition;
  readonly color?: string;
}

export function Icon({ color, ...props }: Props) {
  const [resolvedColor] = useToken("colors", color ? [color] : []);
  return <FontAwesomeIcon {...props} color={resolvedColor || color} />;
}
