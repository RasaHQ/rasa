import { Tooltip as ChakraTooltip, Portal, useToken } from "@chakra-ui/react";
import React, { useState } from "react";
import { useElementDimensionsEffect } from "./hooks/useElementDimensions";

export interface TooltipProps extends ChakraTooltip.RootProps {
  content: React.ReactNode;
  showArrow?: boolean;
  portalled?: boolean;
  portalRef?: React.RefObject<HTMLElement>;
  contentProps?: ChakraTooltip.ContentProps;
  disabled?: boolean;
  contentRef?: React.RefObject<HTMLDivElement | null>;
  onlyShowIfTruncated?: boolean;
  bgColor?: string;
}

export const Tooltip = React.forwardRef<HTMLDivElement, TooltipProps>(
  function Tooltip(props, _ref) {
    const {
      showArrow,
      children,
      disabled,
      portalled = true,
      content,
      contentProps,
      portalRef,
      contentRef,
      onlyShowIfTruncated = false,
      bgColor = "rasawebDeepPurple.900",
      ...rest
    } = props;
    const [isTruncated, setIsTruncated] = useState(false);
    const [resolvedBgColor] = useToken("colors", [bgColor]);

    useElementDimensionsEffect((element) => {
      if (!element || !onlyShowIfTruncated) return;
      if (element.scrollWidth > element.clientWidth) {
        setIsTruncated(true);
      }
    }, contentRef);

    if (disabled || (onlyShowIfTruncated && contentRef && !isTruncated))
      return children;

    return (
      <ChakraTooltip.Root {...rest} openDelay={500}>
        <ChakraTooltip.Trigger asChild>{children}</ChakraTooltip.Trigger>
        <Portal disabled={!portalled} container={portalRef}>
          <ChakraTooltip.Positioner>
            <ChakraTooltip.Content
              ref={contentRef}
              bg={bgColor}
              fontSize="sm"
              px={4}
              py={3}
              borderRadius="md"
              boxShadow="lg"
              fontWeight="normal"
              {...contentProps}
            >
              {showArrow && (
                <ChakraTooltip.Arrow>
                  <ChakraTooltip.ArrowTip
                    style={
                      {
                        borderColor: resolvedBgColor,
                        backgroundColor: resolvedBgColor,
                      } as React.CSSProperties
                    }
                  />
                </ChakraTooltip.Arrow>
              )}
              {content}
            </ChakraTooltip.Content>
          </ChakraTooltip.Positioner>
        </Portal>
      </ChakraTooltip.Root>
    );
  },
);

