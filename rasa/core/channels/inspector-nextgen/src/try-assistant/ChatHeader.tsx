import { Box, Flex, IconButton, Popover, Text } from "@chakra-ui/react";
import { ArrowToBottom, Comment, FileCheck, Icon, Refresh } from "../Icon";
import { OnboardingTooltip } from "../OnboardingTooltip";
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
        <Popover.Root positioning={{ placement: "bottom-end" }}>
          <Popover.Trigger asChild>
            <IconButton
              aria-label="Download"
              data-testid="download-button"
              variant="subtle"
              colorPalette="dark"
              size="sm"
              disabled={!hasEvents}
              _hover={{ bg: "rasaNeutral.200" }}
            >
              <Icon icon={ArrowToBottom} />
            </IconButton>
          </Popover.Trigger>
          <Popover.Positioner>
            <Popover.Content
              width="240px"
              borderRadius="0.5rem"
              boxShadow="0px 2px 10px 0px rgba(0, 0, 0, 0.18)"
              padding="0.5rem"
            >
              <Popover.Body padding="0">
                <Box px="0.75rem" py="0.5rem" textAlign="start">
                  <Text
                    fontWeight="500"
                    fontSize="0.875rem"
                    letterSpacing="0.4px"
                    color="rasawebDeepPurple.800"
                  >
                    Download:
                  </Text>
                </Box>
                <DownloadMenuItem
                  icon={FileCheck}
                  label="E2E tests"
                  onClick={() => downloadE2eTests(conversationList, sessionId)}
                  testId="download-e2e"
                />
                <DownloadMenuItem
                  icon={Comment}
                  label="Conversation"
                  onClick={() =>
                    downloadConversation(conversationList, sessionId)
                  }
                  testId="download-conversation"
                />
              </Popover.Body>
            </Popover.Content>
          </Popover.Positioner>
        </Popover.Root>
        <Tooltip content="Restart conversation" showArrow>
          <IconButton
            aria-label="Restart conversation"
            data-testid="restart-conversation"
            variant="subtle"
            colorPalette="dark"
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

function DownloadMenuItem({
  icon,
  label,
  onClick,
  testId,
}: {
  icon: Parameters<typeof Icon>[0]["icon"];
  label: string;
  onClick: () => void;
  testId: string;
}) {
  return (
    <Box
      as="button"
      display="flex"
      alignItems="center"
      gap="0.625rem"
      width="100%"
      px="0.75rem"
      py="0.5rem"
      borderRadius="0.25rem"
      cursor="pointer"
      _hover={{ bg: "rasaNeutral.100" }}
      onClick={onClick}
      data-testid={testId}
    >
      <Icon icon={icon} />
      <Text
        fontSize="0.875rem"
        letterSpacing="0.4px"
        color="rasawebDeepPurple.800"
      >
        {label}
      </Text>
    </Box>
  );
}
