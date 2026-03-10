import { Button, Flex } from "@chakra-ui/react";
import type { IconDefinition } from "@fortawesome/fontawesome-svg-core";
import { ButtonLink } from "../../../ButtonLink";
import { Icon } from "../../../Icon";

export interface Item {
  title: string;
  payload: string;
  type: "button" | "link" | "attachment";
  icon?: IconDefinition;
}

export const PayloadResponse = ({
  items,
  onQuickReply,
  isDisabled = false,
}: {
  items: Item[];
  onQuickReply?: (payload: string) => void;
  isDisabled?: boolean;
}) => {
  const containerSx = {
    mt: "0.5rem",
    gap: "0.5rem",
    flexWrap: "wrap",
  };

  const buttonSx = {
    whiteSpace: "unset",
    height: "auto",
    minHeight: "2rem",
  };

  const onClick = (e: React.MouseEvent<HTMLButtonElement>, payload: string) => {
    if (!isDisabled) {
      onQuickReply?.(payload);
      e.stopPropagation();
    }
  };

  const mappedButtons = items.map((item) => {
    const isLink = item.type === "link";
    const isButton = item.type === "button";
    const isAttachment = item.type === "attachment";

    if (isLink || isAttachment) {
      return (
        <ButtonLink
          variant="outline"
          colorPalette="dark"
          key={item.title}
          to={item.payload}
          target={"_blank"}
          size="sm"
          css={buttonSx}
          onClick={(e) => e.stopPropagation()}
        >
          {item.title}
          {item.icon && <Icon icon={item.icon} />}
        </ButtonLink>
      );
    }

    if (isButton) {
      return (
        <Button
          variant="outline"
          colorPalette="dark"
          key={item.title}
          onClick={(e) => onClick(e, item.payload)}
          disabled={isDisabled}
          size="sm"
          css={buttonSx}
        >
          {item.title}
        </Button>
      );
    }

    return null;
  });

  return <Flex css={containerSx}>{mappedButtons}</Flex>;
};
