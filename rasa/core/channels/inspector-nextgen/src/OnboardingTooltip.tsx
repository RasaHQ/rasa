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
    <VStack align="start" gap={3} maxW="320px">
      <VStack align="start" gap={2}>
        <Heading size="lg" color="rasawebNeutral.50">
          {onboardingTooltip.title}
        </Heading>
        <Text fontSize="xs" color="rasawebNeutral.50" lineHeight="1.4">
          {onboardingTooltip.description}
        </Text>
      </VStack>
      <HStack justify="space-between" width="100%">
        <HStack gap={2}>
          {onboardingTooltip.counter && (
            <Text fontSize="xs" color="rasawebNeutral.50">
              {onboardingTooltip.counter}
            </Text>
          )}
          <Button
            size="xs"
            variant="ghost"
            color="rasawebNeutral.50"
            _hover={{ bg: "whiteAlpha.200" }}
            onClick={() => onboardingTooltip.onDismiss()}
          >
            Skip tour
          </Button>
        </HStack>
        <Button
          variant="outline"
          colorPalette="dark"
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
