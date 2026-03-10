import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { getBotData } from "../api";
import type { Flow, BotData } from "../types";
import { createFlowNodesFromFlowSteps } from "../utils/flow-import";

export const useConversationData = (
  projectUrl: string,
  botDataEndpoint: string,
) => {
  const {
    data: botData,
    isLoading,
    error,
    refetch,
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

  return {
    flows,
    isLoading,
    error,
    refetch,
  };
};
