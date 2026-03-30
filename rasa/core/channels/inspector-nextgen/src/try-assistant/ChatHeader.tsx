import { Box, Flex, IconButton } from "@chakra-ui/react";
import { ArrowToBottom, Comment, FileCheck, Icon, Refresh } from "../Icon";
import { OnboardingTooltip } from "../OnboardingTooltip";
import { PopoverMenu } from "../PopoverMenu";
import { PopoverMenuItem } from "../PopoverMenuItem";
import { SwitchButton } from "../SwitchButton";
import { Tooltip } from "../Tooltip";
import type { Conversation } from "../types";
import { downloadConversation, downloadE2eTests } from "../utils";

interface ChatHeaderProps {
  onNewConversation: () => void;
  flowView: boolean;
  setFlowView: (flowView: boolean) => void;
  conversationList: Conversation[];
  sessionId: string;
}

export const ChatHeader = ({
  onNewConversation,
  flowView,
  setFlowView,
  conversationList,
  sessionId,
}: ChatHeaderProps) => {
  const hasEvents = conversationList.some(
    (c) => c.totalNumberOfUserMessages > 0,
  );

  return (
    <Flex justifyContent="space-between" flexShrink={1} alignItems="center">
      <OnboardingTooltip target="inspectToggle">
        <SwitchButton
          item={{ value: "inspect", label: "Inspect" }}
          onActivate={() => setFlowView(!flowView)}
          isActive={flowView}
          aria-label="Toggle inspect"
          data-testid="inspect-toggle"
        />
      </OnboardingTooltip>
      <Box display="flex" gap="0.25rem" alignItems="center">
        <PopoverMenu
          trigger={
            <IconButton
              aria-label="Download"
              data-testid="download-button"
              variant="solid"
              colorPalette="light"
              size="sm"
              disabled={!hasEvents}
            >
              <Icon icon={ArrowToBottom} />
            </IconButton>
          }
          header="Download:"
          tooltipContent="Download conversation data"
        >
          <PopoverMenuItem
            icon={FileCheck}
            label="E2E tests"
            onClick={() => downloadE2eTests(conversationList, sessionId)}
            testId="download-e2e"
          />
          <PopoverMenuItem
            icon={Comment}
            label="Conversation"
            onClick={() => downloadConversation(conversationList, sessionId)}
            testId="download-conversation"
          />
        </PopoverMenu>
        <Tooltip content="Restart conversation" showArrow>
          <IconButton
            aria-label="Restart conversation"
            data-testid="restart-conversation"
            variant="solid"
            colorPalette="light"
            onClick={onNewConversation}
            size="sm"
          >
            <Icon icon={Refresh} />
          </IconButton>
        </Tooltip>
      </Box>
    </Flex>
  );
};
