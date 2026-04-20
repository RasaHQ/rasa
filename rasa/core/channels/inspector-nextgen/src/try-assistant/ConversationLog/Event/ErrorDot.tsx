import { Box } from "@chakra-ui/react";
import { Circle, Icon } from "../../../Icon";

export const ErrorDot = () => (
  <Box as="span" flexShrink={0} ml="2" display="inline-flex" data-testid="error-dot">
    <Icon icon={Circle} color="rasaRed.800" size="2xs" />
  </Box>
);
