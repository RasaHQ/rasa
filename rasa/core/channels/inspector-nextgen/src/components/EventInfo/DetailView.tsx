import { Box, Flex, Heading, IconButton } from "@chakra-ui/react";
import {
  ScrollContainer,
  ScrollContent,
  ScrollFixedHeader,
} from "../../VerticalScroll";
import { Icon, XMark } from "../../Icon";

interface DetailViewProps {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

export function DetailView({
  title,
  onClose,
  children,
}: Readonly<DetailViewProps>) {
  return (
    <Box position="relative" height="100%">
      <ScrollContainer>
        <ScrollFixedHeader>
          <Flex
            justifyContent="space-between"
            alignItems="center"
            height="12"
            px="6"
          >
            <Heading textStyle="sm">
              {title}
            </Heading>
            <IconButton
              data-testid="event-details-close"
              variant="ghost"
              size="sm"
              aria-label="Close"
              onClick={onClose}
            >
              <Icon icon={XMark} />
            </IconButton>
          </Flex>
        </ScrollFixedHeader>

        <ScrollContent withSpacing={false} css={{ px: "6", py: "2" }}>
          {children}
        </ScrollContent>
      </ScrollContainer>
      <Box
        position="absolute"
        bottom={0}
        left={0}
        right={0}
        height="6"
        background="linear-gradient(to bottom, transparent, var(--app-colors-bg))"
        pointerEvents="none"
        zIndex={1}
      />
    </Box>
  );
}
