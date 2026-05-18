import { Accordion, Heading } from "@chakra-ui/react";

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
      borderColor="border"
      py="1"
    >
      <Accordion.ItemTrigger colorPalette="gray" px="0" cursor="pointer">
        <Heading flex="1" textStyle="sm" textAlign="left">
          {title}
        </Heading>
        <Accordion.ItemIndicator />
      </Accordion.ItemTrigger>
      <Accordion.ItemContent>
        <Accordion.ItemBody px="0">{children}</Accordion.ItemBody>
      </Accordion.ItemContent>
    </Accordion.Item>
  );
}
