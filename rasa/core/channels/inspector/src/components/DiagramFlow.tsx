import { Box, Button, Flex, Heading, Text } from "@chakra-ui/react";
import mermaid from "mermaid";
import { useOurTheme } from "../theme";
import { formatFlow } from "../helpers/formatters";
import { useEffect, useRef, useState } from "react";
import { Flow, Slot, Stack } from "../types";
import { NoActiveFlow } from "./NoActiveFlow";

interface Props {
  stackFrame?: Stack;
  flows: Flow[];
  slots: Slot[];
}

export const DiagramFlow = (props: Props) => {
  const { rasaSpace } = useOurTheme();
  const mermaidRef = useRef<HTMLPreElement>(null);
  const [text, setText] = useState<string>("");
  const { stackFrame, flows, slots } = props;

  const activeFlowId = stackFrame?.flow_id;
  const activeStepId = stackFrame?.step_id;
  const flow = flows.find(({ id }) => id === activeFlowId);

  const config = {
    startOnLoad: true,
    logLevel: 'info',
    flowchart: {
      useMaxWidth: false,
    },
  }

  useEffect(() => {
    mermaid.mermaidAPI.initialize(config);
  }, []);

  useEffect(() => {
    if (!text) return;
    // mermaid needs to be reloaded every time the text changes but a data-processed
    // attribute prevents it. We need to remove it each time `text` changes
    mermaidRef.current?.removeAttribute("data-processed");
    mermaid.contentLoaded();

    setTimeout(() => {
      const active = document.querySelectorAll(".active")[0];
      if (active) {
        active.scrollIntoView({ behavior: "smooth"});
      }
    }, 0);
  }, [text]);

  useEffect(() => {
    setText(formatFlow(slots, stackFrame, flow, activeStepId));
  }, [text, flow, slots, stackFrame, activeStepId]);

  const handleRestartConversation = () => {
    location.reload();
  };

  const scrollSx = {
    height: "100%",
    overflow: "auto",
    width: "100%",
    textAlign: "center",
    flexDirection: "column",
  };
  const preSx = {
    svg: {
      margin: "0 auto",
    },
  };

  return (
    <Flex direction="column" height="100%">
      <Heading size="lg">
        Flow
        {flow ? (
          <Text as="span" fontWeight="normal">
            {":"} {flow.id}
          </Text>
        ) : null}
      </Heading>
      <Box flexGrow={1} my={rasaSpace[1]} overflow="hidden">
        <Flex sx={scrollSx}>
          {text ? (
            <Box
              as="pre"
              ref={mermaidRef}
              className="mermaid"
              sx={preSx}
              id="mermaid"
            >
              {text}
            </Box>
          ) : (
            <NoActiveFlow />
          )}
        </Flex>
      </Box>
      <Flex justifyContent="space-between" alignItems="flex-end">
        <Button variant="outline" size="sm" onClick={handleRestartConversation}>
          Restart conversation
        </Button>
      </Flex>
    </Flex>
  );
};
