import { forwardRef, useCallback, type Ref } from "react";
import { UtteranceType, type MessagePropsWithTopIntentName } from "../../../types";
import { BotMessage } from "./BotMessage";
import { UserMessage } from "./UserMessage";

export const Message = forwardRef(
  (props: MessagePropsWithTopIntentName, ref: Ref<HTMLDivElement | null>) => {
    const {
      utterance,
      selectable = false,
      isSelected = false,
      isInteractive = false,
      onMessageSelect,
      onQuickReply = undefined,
      conversationEventActions,
      ...otherProps
    } = props;
    const isUser = utterance.type === UtteranceType.User;

    const handleQuickReply = useCallback(
      (payload: string) => {
        if (isInteractive) {
          onQuickReply?.(payload);
        }
      },
      [isInteractive, onQuickReply],
    );

    const handleSelect = () => {
      if (selectable && onMessageSelect) {
        onMessageSelect({ ...utterance, entities: [] });
      }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === "Enter" && selectable && onMessageSelect) {
        onMessageSelect({ ...utterance, entities: [] });
      }
    };

    if (isUser) {
      return (
        <UserMessage
          utterance={utterance}
          {...otherProps}
          isInteractive={isInteractive}
          ref={ref}
          onClick={handleSelect}
          onKeyDown={handleKeyDown}
          tabIndex={0}
          isSelected={selectable && isSelected}
          conversationEventActions={conversationEventActions}
        />
      );
    }
    return (
      <BotMessage
        utterance={utterance}
        {...otherProps}
        isInteractive={isInteractive}
        ref={ref}
        onClick={handleSelect}
        tabIndex={0}
        isSelected={selectable && isSelected}
        onQuickReply={handleQuickReply}
        onKeyDown={handleKeyDown}
        conversationEventActions={conversationEventActions}
      />
    );
  },
);
