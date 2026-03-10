import { Flex, Heading, IconButton } from "@chakra-ui/react";
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
    <ScrollContainer>
      <ScrollFixedHeader>
        <Flex
          justifyContent="space-between"
          alignItems="center"
          borderBottom="1px solid"
          borderColor="rasaNeutral.300"
          px="1.5rem"
          py="0.75rem"
        >
          <Heading fontSize="0.875rem" fontWeight="bold">
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

      <ScrollContent withSpacing={false} css={{ px: "1.5rem", py: "0.5rem" }}>
        {children}
      </ScrollContent>
    </ScrollContainer>
  );
}
