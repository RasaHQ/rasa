import { Box, Flex } from "@chakra-ui/react";
import { useCallback, useEffect } from "react";
import { useConversationData } from "../hooks/useConversationData";
import type { UnionEventType } from "../types";
import { InspectorView } from "../types/inspector";
import {
  useInspectorStore,
  selectStackToShow,
  selectConversationForSelectedElement,
  toggleSelectedElement,
  clearSelectedElement,
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

  useConversationData();

  useEffect(() => {
    onInspectModeChange?.(flowView);
  }, [flowView, onInspectModeChange]);

  const handleSelect = useCallback(
    (selection: UnionEventType) => toggleSelectedElement(selection),
    [],
  );

  const separatorColor = "rasaNeutral.400";
  const canvasSx = {
    borderLeft: "1px solid",
    borderColor: separatorColor,
  };

  return (
    <Flex flexGrow="1" data-testid="try-assistant-container">
      <Box flexBasis={flowView ? "50%" : "100%"}>
        <ChatSection handleSelect={handleSelect} />
      </Box>
      {flowView && (
        <Box
          css={canvasSx}
          flexBasis="calc(50% + 1rem)"
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
              {inspectorView === InspectorView.ActiveFlow && (
                <FlowSection
                  stackToShow={stackToShow}
                  conversationList={conversationList}
                  conversationForSelectedElement={
                    conversationForSelectedElement
                  }
                  flows={flows}
                  flowsLoading={flowsLoading}
                  flowsError={flowsError}
                />
              )}
              {inspectorView === InspectorView.History && <HistorySection />}
              {inspectorView === InspectorView.Memory && <MemorySection />}
            </>
          )}
        </Box>
      )}
    </Flex>
  );
}
