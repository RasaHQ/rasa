import { IconButton, Input, InputGroup } from "@chakra-ui/react";
import {
  type ChangeEvent,
  type ForwardedRef,
  forwardRef,
  type KeyboardEvent,
  useMemo,
  useState,
} from "react";
import { Icon, PaperPlaneTop } from "../Icon";
import { useVoiceCall } from "../hooks/useVoiceCall";
import { VoiceButton } from "./VoiceButton";

interface Props {
  onSubmit: (message: string) => void;
  isDisabled?: boolean;
  startVoiceStreaming: () => Promise<void>;
  stopVoiceStreaming: () => Promise<void>;
  voiceFeaturesEnabled: boolean;
}

export const MessageInput = forwardRef<HTMLInputElement, Props>(
  ({ onSubmit, isDisabled = false, startVoiceStreaming, stopVoiceStreaming, voiceFeaturesEnabled }, ref: ForwardedRef<HTMLInputElement>) => {
    const { callDuration, startVoiceCall, stopVoiceCall, voiceCallState } =
      useVoiceCall({ startVoiceStreaming, stopVoiceStreaming, voiceFeaturesEnabled });
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

    const handleSubmit = () => {
      if (message && !isDisabled) {
        onSubmit(message);
        setMessage("");
      }
    };

    const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
      setMessage(event.target.value);
    };

    const handleKeyboardEvent = (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === "Enter" && !isDisabled) {
        handleSubmit();
      }
    };

    const containerSx = {
      bg: "rasaNeutral.100",
      borderRadius: "3xl",
      display: "flex",
      justifyContent: "center",
    };

    if (voiceCallState === "connecting" || voiceCallState === "active") {
      containerSx.bg = "rasawebNeutral.50";
    }

    return (
      <InputGroup
        onKeyDown={handleKeyboardEvent}
        css={containerSx}
        m="1.5rem"
        px="1rem"
        width={`calc(100% - 3rem)`}
        height="3.5rem"
        data-testid="assistant-input"
      >
        <>
          <Input
            type="text"
            placeholder={placeholder}
            value={message}
            onChange={handleChange}
            disabled={isDisabled || voiceCallState !== "inactive"}
            ref={ref}
            css={{
              borderWidth: "0",
              outline: "none",
              "&:hover": { outline: "none" },
            }}
          />
          {voiceFeaturesEnabled && message.length === 0 ? (
            <VoiceButton
              voiceCallState={voiceCallState}
              startCall={startVoiceCall}
              stopCall={stopVoiceCall}
              isDisabled={isDisabled}
            />
          ) : (
            <IconButton
              variant="solid"
              colorPalette="dark"
              rounded="full"
              aria-label="Send message"
              size="xs"
              fontSize="1rem"
              onClick={handleSubmit}
              disabled={message.length === 0 || isDisabled}
            >
              <Icon icon={PaperPlaneTop} />
            </IconButton>
          )}
        </>
      </InputGroup>
    );
  },
);
