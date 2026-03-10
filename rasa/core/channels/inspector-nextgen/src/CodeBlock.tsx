import { Box, Button, HStack, Text } from "@chakra-ui/react";
import { useEffect, useRef, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Check, Clone, Icon } from "./Icon";
import { useInspectorContext } from "./InspectorContext";
import { Tooltip } from "./Tooltip";

interface CodeBlockProps {
  language?: string;
  code: string;
  hideHeader?: boolean;
}

export const CodeBlock = ({
  language,
  code,
  hideHeader = false,
}: CodeBlockProps) => {
  const { logError } = useInspectorContext();
  const [hasCopied, setHasCopied] = useState(false);
  const timeoutRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setHasCopied(true);

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = globalThis.setTimeout(() => {
        setHasCopied(false);
      }, 2000) as unknown as number;
    } catch (error) {
      logError(error, {
        tags: {
          component: "CodeBlock",
          action: "handleCopy",
        },
      });
    }
  };

  return (
    <Box
      bg="rasawebNeutral.900"
      borderRadius="0.5rem"
      overflow="hidden"
      my="1rem"
    >
      {hideHeader ? null : (
        <HStack
          bg="rasawebNeutral.700"
          px="1rem"
          py="0.5rem"
          justify="space-between"
        >
          <Text
            size="xs"
            fontFamily="IBM Plex Mono, monospace"
            color="rasaNeutral.100"
          >
            {language || "code"}
          </Text>
          <Tooltip content="Copy" showArrow>
            <Button // TODO: figure this out on the design system (same as "Skip tour" in OnboardingTooltip)
              size="sm"
              onClick={() => void handleCopy()}
              variant="ghost"
              color="rasaNeutral.50"
              aria-label="Copy"
              _hover={{
                bg: hasCopied ? "none" : "whiteAlpha.200",
                cursor: hasCopied ? "default" : "pointer",
              }}
              height="1.5rem"
              width={hasCopied ? "auto" : "1.5rem"}
              gap="0.25rem"
              px={hasCopied ? "0.5rem" : "0"}
              // needs explicit style to override the bg value applied by Tooltip
              // TODO: fix in the design system
              bg="transparent"
            >
              {hasCopied ? <Icon icon={Check} /> : <Icon icon={Clone} />}
              {hasCopied ? (
                <Text size="sm" color="rasaNeutral.100">
                  Copied!
                </Text>
              ) : null}
            </Button>
          </Tooltip>
        </HStack>
      )}
      <SyntaxHighlighter
        language={language}
        style={vscDarkPlus}
        customStyle={{
          margin: 0,
          padding: "1rem",
          fontFamily: "IBM Plex Mono, monospace",
          fontSize: "0.813rem",
          backgroundColor: "inherit",
        }}
        codeTagProps={{
          style: {
            fontFamily: "IBM Plex Mono, monospace",
          },
        }}
      >
        {code}
      </SyntaxHighlighter>
    </Box>
  );
};

