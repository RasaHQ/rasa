import { useInspectorContext } from "../InspectorContext";
import {
  PlaceholderImage,
  type Conversation,
  type Flow,
  type Stack,
} from "../types";
import { NoData } from "../Placeholder";
import { Flow as FlowComponent } from "./Flow";

interface FlowSectionProps {
  stackToShow: Stack | undefined;
  conversationList: Conversation[];
  conversationForSelectedElement: Conversation | undefined;
  flows: Flow[];
  flowsLoading: boolean;
  flowsError: Error | null;
}

export const FlowSection = ({
  stackToShow,
  conversationList,
  conversationForSelectedElement,
  flows,
  flowsLoading,
  flowsError,
}: FlowSectionProps) => {
  const { logError } = useInspectorContext();

  const showFlow = stackToShow?.flowId
    ? flows?.find((flow: Flow) => flow.id === stackToShow?.flowId)
    : undefined;

  if (flowsLoading) {
    return (
      <NoData
        image={PlaceholderImage.NoFlow}
        shortLabel="Loading flows..."
        longLabel="Please wait while we load the conversation flows"
      />
    );
  }

  if (flowsError) {
    logError(flowsError, {
      tags: {
        component: "TryAssistant",
        action: "getConversationData",
      },
    });
    return (
      <NoData
        image={PlaceholderImage.NoFlow}
        shortLabel="Failed to load flows"
        longLabel={"Error while obtaining trained flow data"}
      />
    );
  }

  if (!stackToShow?.flowId) {
    return (
      <NoData
        image={PlaceholderImage.NoFlow}
        shortLabel="No flow is currently active"
        longLabel="Type a message to talk to the latest trained version of your assistant"
      />
    );
  }

  if (!showFlow) {
    return (
      <NoData
        image={PlaceholderImage.NoFlow}
        shortLabel="This flow isn't available anymore"
        longLabel={`"${stackToShow?.flowId}" was removed, so it can't be previewed in this version`}
      />
    );
  }

  return (
    <FlowComponent
      flow={showFlow}
      selectedNodeId={stackToShow?.stepId}
      conversation={
        conversationForSelectedElement ||
        conversationList[conversationList.length - 1]
      }
    />
  );
};
