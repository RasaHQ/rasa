import { Flex, Heading, HStack, Text } from "@chakra-ui/react";
import { InspectorViewPopover } from "./InspectorViewPopover";

interface Props {
  title: string;
  text?: string;
  sticky?: boolean;
  showViewSwitcher?: boolean;
}

export const InspectorViewHeader = ({
  title,
  text,
  sticky,
  showViewSwitcher = true,
}: Props) => {
  const stickyStyles = {
    position: "absolute",
    top: "0",
    left: "0",
    right: "0",
  };
  return (
    <Flex
      height="12"
      flexShrink={0}
      px="6"
      alignItems="center"
      justifyContent="space-between"
      css={sticky ? stickyStyles : {}}
    >
      <HStack minWidth={0}>
        <Heading textStyle="sm">{title}</Heading>
        {text && (
          <Text
            textStyle="sm"
            overflow="hidden"
            textOverflow="ellipsis"
            whiteSpace="nowrap"
          >
            {text}
          </Text>
        )}
      </HStack>
      {showViewSwitcher && <InspectorViewPopover />}
    </Flex>
  );
};
