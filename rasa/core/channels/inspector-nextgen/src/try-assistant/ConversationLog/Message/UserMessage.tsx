import { type FlexProps, Text } from "@chakra-ui/react";
import { forwardRef } from "react";
import { isInternalRasaMessage } from "../../../utils";
import type { MessagePropsWithTopIntentName } from "../../../types";
import { MessageMarkup } from "./MessageMarkup";

export const UserMessage = forwardRef<
  HTMLDivElement | null,
  MessagePropsWithTopIntentName
>((props: MessagePropsWithTopIntentName, ref) => {
  const { utterance, ...otherProps } = props;

  const containerSxUser: FlexProps = {
    flexDirection: "row-reverse",
    textAlign: "right",
  };

  const messageSxUser = {
    p: "4",
    cursor: props.selectable ? "pointer" : "auto",
    whiteSpace: "pre-wrap",
  };

  if (isInternalRasaMessage(utterance.text || "")) {
    return null;
  }

  return (
    <MessageMarkup
      utterance={utterance}
      containerSx={containerSxUser}
      messageSx={messageSxUser}
      {...otherProps}
      isUser={true}
      ref={ref}
    >
      <Text textStyle="sm">
        {utterance.text}
      </Text>
    </MessageMarkup>
  );
});
