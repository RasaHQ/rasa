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
            height="3rem"
            bg="rasaNeutral.50"
            px="1.5rem"
          >
            <Heading size="md">
              {title}
            </Heading>
            <IconButton
              data-testid="event-details-close"
              variant="solid"
              colorPalette="light"
              size="sm"
              aria-label="Close"
              onClick={onClose}
            >
              <Icon icon={XMark} />
            </IconButton>
          </Flex>
        </ScrollFixedHeader>

        <ScrollContent withSpacing={false} css={{ px: "1.5rem", py: "0.5rem" }}>
          {children}
        </ScrollContent>
      </ScrollContainer>
      <Box
        position="absolute"
        bottom={0}
        left={0}
        right={0}
        height="24px"
        background="linear-gradient(to bottom, transparent, white)"
        pointerEvents="none"
        zIndex={1}
      />
    </Box>
  );
}
