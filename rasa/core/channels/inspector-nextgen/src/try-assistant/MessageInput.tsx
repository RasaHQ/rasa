import { IconButton, Input, InputGroup } from "@chakra-ui/react";
import {
  type ChangeEvent,
  type ForwardedRef,
  forwardRef,
  type KeyboardEvent,
  type RefObject,
  useCallback,
  useMemo,
  useState,
} from "react";
import { Icon, PaperPlaneTop } from "../Icon";
import { useVoiceCall } from "../hooks/useVoiceCall";
import type { VoiceErrorHandler } from "../types";
import { VoiceButton } from "./VoiceButton";

interface Props {
  onSubmit: (message: string) => void;
  isDisabled?: boolean;
  startVoiceStreaming: () => Promise<void>;
  stopVoiceStreaming: () => Promise<void>;
  onVoiceErrorRef: RefObject<VoiceErrorHandler>;
  voiceFeaturesEnabled: boolean;
}

export const MessageInput = forwardRef<HTMLInputElement, Props>(
  ({ onSubmit, isDisabled = false, startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled }, ref: ForwardedRef<HTMLInputElement>) => {
    const { callDuration, startVoiceCall, stopVoiceCall, voiceCallState } =
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, onVoiceErrorRef, voiceFeaturesEnabled });
    const [message, setMessage] = useState("");
    const placeholder = useMemo(() => {
      if (voiceCallState === "connecting") {
        return "Connecting...";
      }
      if (voiceCallState === "active") {
        return `Voice conversation in progress (${callDuration})`;
      }
      return "Type your message";
    }, [voiceCallState, callDuration]);

    const handleSubmit = useCallback(() => {
      if (message && !isDisabled) {
        onSubmit(message);
        setMessage("");
      }
    }, [onSubmit, isDisabled, setMessage, message]);

    const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
      setMessage(event.target.value);
    };

    const handleKeyboardEvent = (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "Enter" && !isDisabled) {
        handleSubmit();
      }
    };

    const button = useCallback(() => {
      if (voiceFeaturesEnabled && message.length === 0) {
        return <VoiceButton
          voiceCallState={voiceCallState}
          startCall={startVoiceCall}
          stopCall={stopVoiceCall}
          isDisabled={isDisabled}
        />;
      }
      return <IconButton
        variant="solid"
        colorPalette="purple"
        rounded="full"
        aria-label="Send message"
        size="xs"
        onClick={handleSubmit}
        disabled={message.length === 0 || isDisabled}
      >
        <Icon icon={PaperPlaneTop} />
      </IconButton>
    }, [voiceFeaturesEnabled, message, handleSubmit, isDisabled, startVoiceCall, stopVoiceCall, voiceCallState]);

    return (
      <InputGroup
        onKeyDown={handleKeyboardEvent}
        data-testid="assistant-input"
        endElement={button()}
      >
        <Input
          type="text"
          size="3xl"
          bg="bg.subtle"
          placeholder={placeholder}
          value={message}
          onChange={handleChange}
          disabled={isDisabled || voiceCallState !== "inactive"}
          ref={ref}
        />
      </InputGroup>
    );
  },
);
