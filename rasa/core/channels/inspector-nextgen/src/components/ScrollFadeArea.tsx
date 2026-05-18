/* istanbul ignore file */
import { Box, type BoxProps } from "@chakra-ui/react";

const FADE_HEIGHT = "6";

interface ScrollFadeAreaProps extends BoxProps {
  children: React.ReactNode;
  fadeBg?: string;
}

export const ScrollFadeArea = ({
  children,
  fadeBg = "var(--app-colors-bg)",
  ...props
}: ScrollFadeAreaProps) => (
  <Box position="relative" flex={1} overflow="hidden" {...props}>
    <Box position="absolute" inset={0} overflowY="auto">
      {children}
    </Box>
    <Box
      position="absolute"
      bottom={0}
      left={0}
      right={0}
      height={FADE_HEIGHT}
      background={`linear-gradient(to bottom, transparent, ${fadeBg})`}
      pointerEvents="none"
      zIndex={1}
    />
  </Box>
);
