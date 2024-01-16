import {
  Box,
  Button,
  Flex,
  FlexProps,
  HStack,
  TabList,
  TabPanel,
  TabPanels,
  TabProps,
  Tabs,
  useColorModeValue,
  useTab,
  useToast,
} from "@chakra-ui/react";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { useOurTheme } from "../theme";
import { Event, Slot } from "../types";
import { formatTestCases } from "../helpers/formatters";
import { useCopyToClipboard } from "usehooks-ts";
import { useEffect } from "react";
import { FullscreenButton } from "./FullscreenButton";
import { SlotTable, Slots } from "./Slots";

interface Props extends FlexProps {
  rasaChatSessionId?: string;
  events: Event[];
  story: string;
  slots: Slot[];
}

export const DialougeInformation = ({
  sx,
  rasaChatSessionId = "-",
  events,
  story,
  slots,
  ...props
}: Props) => {
  const toast = useToast();
  const { rasaSpace, rasaFontSizes } = useOurTheme();
  const [copiedText, copyToClipboard] = useCopyToClipboard();

  const containerSx = {
    ...sx,
    p: 0,
    flexDirection: "column",
  };
  const tabsSx = {
    height: "100%",
    display: "flex",
    flexDirection: "column",
  };
  const tabListSx = {
    px: rasaSpace[1],
    py: rasaSpace[0.5],
    borderBottom: "1px solid",
    borderColor: useColorModeValue("neutral.400", "neutral.400"),
  };
  const tabPanelSx = {
    height: "100%",
    overflow: "hidden",
    padding: rasaSpace[0.5],
    position: "relative",
  };
  const overflowBox = {
    height: "100%",
    overflow: "auto",
    padding: rasaSpace[0.5],
  };
  const highlighterSx = {
    background: useColorModeValue("neutral.50", "neutral.50"),
    fontSize: rasaFontSizes.sm,
    letterSpacing: "0",
  };
  const buttonsProps = {
    position: "absolute",
    bg: useColorModeValue("neutral.50", "neutral.50"),
    right: rasaSpace[1],
    bottom: rasaSpace[1],
  };

  useEffect(() => {
    if (toast.isActive("copy-to-clipboard") || !copiedText) return;
    toast({
      id: "copy-to-clipboard",
      title: "Text copied to clipboard",
      status: "success",
      duration: 2000,
      isClosable: true,
    });
  }, [copiedText, toast]);

  const testCases = formatTestCases(events, rasaChatSessionId);

  return (
    <Flex sx={containerSx} {...props}>
      <Tabs
        sx={tabsSx}
        size="sm"
        variant="solid-rounded"
        colorScheme="rasaPurple"
      >
        <TabList sx={tabListSx}>
          <CustomTab>Slots</CustomTab>
          <CustomTab>End-2-end test</CustomTab>
          <CustomTab>Tracker state</CustomTab>
        </TabList>
        <TabPanels height="100%" overflow="hidden">
          <TabPanel sx={tabPanelSx}>
            <Box sx={overflowBox}>
              <Slots slots={slots} />
            </Box>
            <HStack spacing={rasaSpace[0.5]} sx={buttonsProps}>
              <FullscreenButton title="Slots">
                <SlotTable slots={slots} />
              </FullscreenButton>
            </HStack>
          </TabPanel>
          <TabPanel sx={tabPanelSx}>
            <Box sx={overflowBox}>
              <SyntaxHighlighter customStyle={highlighterSx}>
                {testCases}
              </SyntaxHighlighter>
            </Box>

            <HStack spacing={rasaSpace[0.5]} sx={buttonsProps}>
              <Button
                size="sm"
                variant="outline"
                onClick={() => copyToClipboard(testCases)}
              >
                Copy
              </Button>
              <FullscreenButton title="End-2-end test">
                <SyntaxHighlighter customStyle={highlighterSx}>
                  {testCases}
                </SyntaxHighlighter>
              </FullscreenButton>
            </HStack>
          </TabPanel>
          <TabPanel sx={tabPanelSx}>
            <Box sx={overflowBox}>
              <SyntaxHighlighter customStyle={highlighterSx}>
                {story}
              </SyntaxHighlighter>
            </Box>
            <HStack spacing={rasaSpace[0.5]} sx={buttonsProps}>
              <Button
                size="sm"
                variant="outline"
                onClick={() => copyToClipboard(story)}
              >
                Copy
              </Button>
              <FullscreenButton title="Tracker state">
                <SyntaxHighlighter customStyle={highlighterSx}>
                  {story}
                </SyntaxHighlighter>
              </FullscreenButton>
            </HStack>
          </TabPanel>
        </TabPanels>
      </Tabs>
    </Flex>
  );
};

const CustomTab = (props: TabProps) => {
  const { rasaRadii } = useOurTheme();
  const tabProps = useTab(props);
  const isSelected = !!tabProps["aria-selected"];
  const variant = isSelected ? "solidRasa" : "ghost";
  const selectedColor = useColorModeValue("neutral.50", "neutral.50");
  const unselectedColor = useColorModeValue("neutral.900", "neutral.900");

  const buttonSx = {
    borderRadius: rasaRadii.full,
    color: isSelected ? selectedColor : unselectedColor,
  };

  return (
    <Button sx={buttonSx} variant={variant} {...tabProps} size="sm">
      {tabProps.children}
    </Button>
  );
};
