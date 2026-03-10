import {
  Box,
  Button,
  Flex,
  Image,
  Separator,
  Text,
  useDisclosure,
} from "@chakra-ui/react";
import { forwardRef, useState } from "react";
import type {
  MessagePropsWithTopIntentName,
  ResponseData,
} from "../../../types";
import { ExternalLink, Icon, Code as IconCode, Paperclip } from "../../../Icon";
import { Modal } from "../../../Modal";
import { ConversationEventActionButton } from "../../ConversationEventActionButton";
import { MessageMarkup } from "./MessageMarkup";
import { type Item as PayloadItem, PayloadResponse } from "./PayloadResponse";
import { CodeBlock } from "../../../CodeBlock";

const startsWithProtocol = (payload: string) =>
  /^\b((mailto|tel|sms):|[a-z]+:\/\/)/i.test(payload);

export const BotMessage = forwardRef<
  HTMLDivElement | null,
  MessagePropsWithTopIntentName
>((props: MessagePropsWithTopIntentName, ref) => {
  const { utterance, onQuickReply, isSelected, isInteractive, conversationEventActions, ...otherProps } =
    props;
  const { open, onOpen, onClose } = useDisclosure();
  const [isHovered, setIsHovered] = useState(false);

  const image = utterance.responseData?.image;
  const buttons = getButtons(utterance.responseData);
  const text = utterance.text;
  const custom = utterance.responseData?.custom;

  const modalActions = [
    {
      id: "Cancel",
      title: "Cancel",
      onClick: (e: React.MouseEvent<HTMLElement>) => {
        e.stopPropagation();
        onClose();
      },
      "aria-label": "Cancel",
      variant: "outline" as const,
      colorPalette: "dark" as const,
    },
  ];

  const imageSx = {
    borderTopRadius: "0.25rem",
    borderTopRightRadius: "1rem",
    objectFit: "cover",
    objectPosition: "center",
    width: "100%",
  };

  const dividerSx = {
    borderColor: "#DDE2EF",
    my: "0.5rem",
  };

  return (
    <MessageMarkup
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      utterance={utterance}
      topOverlay={isHovered ? <ConversationEventActionButton event={utterance} actions={conversationEventActions || []} /> : undefined}
      isSelected={isSelected}
      {...otherProps}
      ref={ref}
      messageBreakout={
        buttons.length ? (
          <PayloadResponse
            items={buttons}
            onQuickReply={onQuickReply}
            isDisabled={!isInteractive}
          />
        ) : null
      }
    >
      {image ? <Image src={image} css={imageSx} alt="Assistant image" /> : null}

      <Box p="1rem" _empty={{ display: "none" }}>
        {text ? (
          <Text size="md" whiteSpace="pre-wrap">
            {text}
          </Text>
        ) : null}

        {text && custom ? (
          <Separator orientation="horizontal" css={dividerSx} />
        ) : null}

        {custom ? (
          <Box>
            <Flex alignItems="center">
              <Icon
                icon={IconCode}
                fontSize="1rem"
                style={{ marginRight: "7" }}
              />
              JSON
            </Flex>
            <Modal
              button={() => (
                <Button size="sm" mt="3" colorPalette="dark">
                  Click to view code
                </Button>
              )}
              title={utterance.metadata?.utter_action}
              actions={modalActions}
              open={open}
              onClose={onClose}
              onOpen={onOpen}
            >
              <CodeBlock
                language="json"
                code={JSON.stringify(custom, null, 2) || "-"}
              />
            </Modal>
          </Box>
        ) : null}
      </Box>
    </MessageMarkup>
  );
});

const getButtons = (responseData?: ResponseData | null): PayloadItem[] => {
  if (!responseData) return [];
  const buttons: PayloadItem[] = [];

  const responseButtons = [
    ...responseData.buttons,
    ...responseData.quickReplies,
  ].filter(Boolean);

  for (const button of responseButtons) {
    const isLink = startsWithProtocol(button.payload);
    buttons.push({
      title: button.title,
      payload: button.payload,
      type: isLink ? "link" : "button",
      ...(isLink ? { icon: ExternalLink } : {}),
    });
  }

  if (responseData.attachment) {
    buttons.push({
      title: responseData.attachment,
      payload: responseData.attachment,
      icon: Paperclip,
      type: "attachment",
    });
  }

  return buttons;
};
