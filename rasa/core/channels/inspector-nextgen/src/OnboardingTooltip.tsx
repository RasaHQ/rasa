import { useRef, type ReactNode } from "react";
import { Button, Heading, HStack, Text, VStack } from "@chakra-ui/react";
import { useInspectorContext } from "./InspectorContext";
import { Tooltip } from "./Tooltip";
import type { OnboardingTooltipTarget } from "./types";

interface Props {
  target: OnboardingTooltipTarget;
  children: ReactNode;
}

export const OnboardingTooltip = ({ target, children }: Props) => {
  const { onboardingTooltips } = useInspectorContext();
  const contentRef = useRef<HTMLDivElement>(null);

  const onboardingTooltip = onboardingTooltips.find((t) => t.target === target);

  if (!onboardingTooltip) return <>{children}</>;

  const tooltipContent = (
    <VStack align="start" gap={3} maxW="80">
      <VStack align="start" gap={2}>
        <Heading textStyle="lg" color="fg.inverted">
          {onboardingTooltip.title}
        </Heading>
        <Text color="fg.inverted">
          {onboardingTooltip.description}
        </Text>
      </VStack>
      <HStack justify="space-between" width="100%">
        <HStack gap={2}>
          {onboardingTooltip.counter && (
            <Text color="fg.inverted">
              {onboardingTooltip.counter}
            </Text>
          )}
          <Button
            size="xs"
            variant="ghost"
            color="fg.inverted"
            _hover={{ bg: "whiteAlpha.200" }}
            onClick={() => onboardingTooltip.onDismiss()}
          >
            Skip tour
          </Button>
        </HStack>
        <Button
          variant="outline"
          colorPalette="gray"
          size="xs"
          onClick={() => onboardingTooltip.onAction()}
        >
          {onboardingTooltip.actionLabel}
        </Button>
      </HStack>
    </VStack>
  );

  return (
    <Tooltip
      content={tooltipContent}
      showArrow
      open={true}
      closeDelay={0}
      interactive={true}
      portalled={true}
      contentProps={{ px: 6, py: 6 }}
      contentRef={contentRef}
    >
      {children}
    </Tooltip>
  );
};
