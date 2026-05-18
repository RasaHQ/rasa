import { Box, Flex, Heading, IconButton } from "@chakra-ui/react";
import { ArrowToBottom, Comment, FileCheck, Icon, Refresh } from "../Icon";
import { OnboardingTooltip } from "../OnboardingTooltip";
import { PopoverMenu } from "../PopoverMenu";
import { PopoverMenuItem } from "../PopoverMenuItem";
import { useInspectorStore } from "../store";
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
  const isEmbedded = useInspectorStore((s) => s.isEmbedded);

  return (
    <Flex justifyContent="space-between" flex={1} alignItems="center" bg="bg.panel">
      {isEmbedded ? (
        <OnboardingTooltip target="inspectToggle">
          <SwitchButton
            item={{ value: "inspect", label: "Inspect" }}
            onActivate={() => setFlowView(!flowView)}
            isActive={flowView}
            aria-label="Toggle inspect"
            data-testid="inspect-toggle"
          />
        </OnboardingTooltip>
      ) : (
        <Heading textStyle="sm">Preview</Heading>
      )}
      <Box display="flex" gap="1" alignItems="center">
        <PopoverMenu
          trigger={
            <IconButton
              aria-label="Download"
              data-testid="download-button"
              variant="ghost"
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
            variant="ghost"
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
