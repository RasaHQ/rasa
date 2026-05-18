import { Box, Flex } from "@chakra-ui/react";
import { useCallback, useEffect } from "react";
import { useConversationData } from "../hooks/useConversationData";
import { useIsLargeScreen } from "../hooks/useIsLargeScreen";
import type { UnionEventType } from "../types";
import { InspectorView } from "../types/inspector";
import {
  useInspectorStore,
  selectStackToShow,
  selectConversationForSelectedElement,
  toggleSelectedElement,
  clearSelectedElement,
  setInspectorView,
} from "../store";
import { ChatSection } from "./ChatSection";
import { EventDetails } from "./EventDetails";
import { FlowSection } from "./FlowSection";
import { HistorySection } from "./HistorySection";
import { MemorySection } from "./MemorySection";

type Props = {
  onInspectModeChange?: (inspectMode: boolean) => void;
};

export function TryAssistant({ onInspectModeChange }: Readonly<Props>) {
  const flowView = useInspectorStore((s) => s.inspectMode);
  const inspectorView = useInspectorStore((s) => s.inspectorView);
  const selectedElement = useInspectorStore((s) => s.selectedElement);
  const conversationList = useInspectorStore((s) => s.conversationList);
  const flows = useInspectorStore((s) => s.flows);
  const flowsLoading = useInspectorStore((s) => s.flowsLoading);
  const flowsError = useInspectorStore((s) => s.flowsError);
  const stackToShow = useInspectorStore(selectStackToShow);
  const conversationForSelectedElement = useInspectorStore(
    selectConversationForSelectedElement,
  );

  const isEmbedded = useInspectorStore((s) => s.isEmbedded);
  const isLargeScreen = useIsLargeScreen();
  const showAllLayout = flowView && isLargeScreen && inspectorView === InspectorView.All;

  const containerProps = isEmbedded
    ? { flexGrow: "1", bg: "bg.panel", borderRadius: "2xl", overflow: "hidden" as const }
    : { height: "100%", bg: "bg.panel", borderRadius: "2xl", overflow: "hidden" as const };

  useConversationData();

  useEffect(() => {
    onInspectModeChange?.(flowView);
  }, [flowView, onInspectModeChange]);

  useEffect(() => {
    if (isLargeScreen && flowView && inspectorView !== InspectorView.All) {
      setInspectorView(InspectorView.All);
    }
    // Only run when inspect mode is toggled on while on a large screen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flowView]);

  useEffect(() => {
    if (!isLargeScreen && inspectorView === InspectorView.All) {
      setInspectorView(InspectorView.ActiveFlow);
    }
  }, [isLargeScreen, inspectorView]);

  const handleSelect = useCallback(
    (selection: UnionEventType) => toggleSelectedElement(selection),
    [],
  );

  const separatorColor = "border.emphasized";
  const panelBorderLeft = {
    borderLeft: "1px solid",
    borderColor: separatorColor,
  };

  const chatPanelStyles = {
    bg: "bg.panel",
    flexBasis: flowView ? "600px" : "100%",
    flexShrink: "1",
    flexGrow: "0",
    maxWidth: flowView ? "600px" : "unset",
  };

  const flowSectionElement = (
    <FlowSection
      stackToShow={stackToShow}
      conversationList={conversationList}
      conversationForSelectedElement={conversationForSelectedElement}
      flows={flows}
      flowsLoading={flowsLoading}
      flowsError={flowsError}
    />
  );

  if (showAllLayout) {
    return (
      <Flex {...containerProps} data-testid="try-assistant-container">
        <Box css={chatPanelStyles} minWidth={0}>
          <ChatSection handleSelect={handleSelect} />
        </Box>

        <Box
          css={panelBorderLeft}
          flex="800px 1"
          data-testid="inspector-canvas"
        >
          {flowSectionElement}
        </Box>

        <Flex
          css={panelBorderLeft}
          flex="600px 0"
          direction="column"
          data-testid="inspector-side-panel"
        >
          {selectedElement ? (
            <EventDetails
              event={selectedElement}
              onClose={clearSelectedElement}
              flows={flows}
            />
          ) : (
            <>
              <Box flex="1" minHeight={0} overflow="hidden">
                <HistorySection />
              </Box>
              <Box
                flex="1"
                minHeight={0}
                overflow="hidden"
                borderTop="1px solid"
                borderColor={separatorColor}
              >
                <MemorySection showViewSwitcher={false} />
              </Box>
            </>
          )}
        </Flex>
      </Flex>
    );
  }

  return (
    <Flex {...containerProps} data-testid="try-assistant-container">
      <Box css={chatPanelStyles}>
        <ChatSection handleSelect={handleSelect} />
      </Box>
      {flowView && (
        <Box
          css={panelBorderLeft}
          flex="1"
          data-testid="inspector-canvas"
        >
          {selectedElement ? (
            <EventDetails
              event={selectedElement}
              onClose={clearSelectedElement}
              flows={flows}
            />
          ) : (
            <>
              {inspectorView === InspectorView.ActiveFlow && flowSectionElement}
              {inspectorView === InspectorView.History && <HistorySection />}
              {inspectorView === InspectorView.Memory && <MemorySection />}
            </>
          )}
        </Box>
      )}
    </Flex>
  );
}
