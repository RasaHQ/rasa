import { Accordion, Text } from "@chakra-ui/react";

interface EventAccordionItemProps {
  title: string;
  value: string;
  children: React.ReactNode;
}

export function EventAccordionItem({
  title,
  value,
  children,
}: EventAccordionItemProps) {
  return (
    <Accordion.Item
      value={value}
      borderBottom="1px solid"
      borderColor="rasaNeutral.300"
      py="0.25rem"
    >
      <Accordion.ItemTrigger px="0" cursor="pointer">
        <Text flex="1" fontWeight="bold" fontSize="0.875rem" textAlign="left">
          {title}
        </Text>
        <Accordion.ItemIndicator />
      </Accordion.ItemTrigger>
      <Accordion.ItemContent>
        <Accordion.ItemBody px="0">{children}</Accordion.ItemBody>
      </Accordion.ItemContent>
    </Accordion.Item>
  );
}
