import { IconButton } from "@chakra-ui/react";
import { Icon, Stop, WaveformLines } from "../Icon";
import { Tooltip } from "../Tooltip";
import { type VoiceCallState } from "../hooks/useVoiceCall";

interface Props {
  voiceCallState: VoiceCallState;
  startCall: () => Promise<void>;
  stopCall: () => Promise<void>;
  isDisabled?: boolean;
}

export const VoiceButton = ({
  voiceCallState,
  startCall,
  stopCall,
  isDisabled,
}: Props) => {
  return voiceCallState === "inactive" || voiceCallState === "connecting" ? (
    <Tooltip
      content="Start voice conversation"
      aria-label="Start voice conversation"
      positioning={{ placement: "top" }}
      bgColor="rasawebDeepPurple.800"
      showArrow
    >
      <IconButton
        variant="solid"
        colorPalette="dark"
        rounded="full"
        aria-label="Start voice conversation"
        size="xs"
        fontSize="1rem"
        onClick={() => void startCall()}
        disabled={isDisabled || voiceCallState === "connecting"}
      >
        <Icon icon={WaveformLines} />
      </IconButton>
    </Tooltip>
  ) : (
    <Tooltip
      content="End conversation"
      aria-label="End conversation"
      positioning={{ placement: "top" }}
      bgColor="rasawebDeepPurple.800"
      showArrow
    >
      <IconButton
        variant="solid"
        colorPalette="dark"
        rounded="full"
        aria-label="Stop voice conversation"
        size="xs"
        fontSize="1rem"
        onClick={() => void stopCall()}
        disabled={isDisabled}
      >
        <Icon icon={Stop} />
      </IconButton>
    </Tooltip>
  );
};
