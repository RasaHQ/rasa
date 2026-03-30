import { Heading, HStack, Text } from "@chakra-ui/react";
import { InspectorViewPopover } from "./InspectorViewPopover";

interface Props {
  title: string;
  text?: string;
  sticky?: boolean;
}

export const InspectorViewHeader = ({ title, text, sticky }: Props) => {
  const stickyStyles = {
    position: "absolute",
    top: "0",
    left: "0",
  };
  return (
    <HStack
      width="100%"
      px="1.5rem"
      py="1rem"
      height="auto"
      flexDirection="row"
      justifyContent="space-between"
      css={sticky ? stickyStyles : {}}
    >
      <HStack>
        <Heading size="md">{title}</Heading>
        {text && (
          <Text
            size="md"
            overflow="hidden"
            textOverflow="ellipsis"
            whiteSpace="nowrap"
          >
            {text}
          </Text>
        )}
      </HStack>
      <InspectorViewPopover />
    </HStack>
  );
};
