import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";
import { getBotData } from "../api";
import type { Flow, BotData } from "../types";
import { createFlowNodesFromFlowSteps } from "../utils/flow-import";
import { inspectorStore, useInspectorStore } from "../store";

export const useConversationData = () => {
  const projectUrl = useInspectorStore((s) => s.projectUrl);
  const botDataEndpoint = useInspectorStore((s) => s.botDataEndpoint);

  const {
    data: botData,
    isLoading,
    error,
  } = useQuery<BotData, Error>({
    queryKey: ["botData", projectUrl],
    queryFn: () => getBotData({ projectUrl, botDataEndpoint }),
    enabled: !!projectUrl,
    refetchOnWindowFocus: false,
  });

  const flows = useMemo(() => {
    if (!botData?.flows) return [];

    const allFlows: Flow[] = [];
    for (const [flowId, flow] of Object.entries(botData?.flows)) {
      const { nodes: newNodes, edges: newEdges } = createFlowNodesFromFlowSteps(
        flow.steps,
        flowId,
        Object.keys(botData?.flows),
      );
      allFlows.push({
        id: flowId,
        name: flow.name || flowId,
        description: flow.description,
        nodes: newNodes,
        edges: newEdges,
      });
    }

    return allFlows;
  }, [botData?.flows]);

  useEffect(() => {
    inspectorStore.setState((prev) => ({
      ...prev,
      flows,
      flowsLoading: isLoading,
      flowsError: error,
    }));
  }, [flows, isLoading, error]);

  useEffect(() => {
    if (botData?.assistant_id !== undefined) {
      inspectorStore.setState((prev) => ({
        ...prev,
        assistantId: botData.assistant_id ?? null,
      }));
    }
  }, [botData?.assistant_id]);
};
