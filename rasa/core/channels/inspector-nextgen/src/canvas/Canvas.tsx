import { Box } from "@chakra-ui/react";
import { ReactFlow } from "reactflow";
import "reactflow/dist/style.css";
import { useCanvasContext } from "../CanvasContext";
import { InspectorViewHeader } from "../components/InspectorViewHeader";
import { DEFAULT_NODE_HEIGHT, DEFAULT_NODE_WIDTH } from "../constants";
import { BottomControls } from "./BottomControls";
import { AddEdge } from "./edges/AddEdge";
import { CustomEdge } from "./edges/CustomEdge";
import { EndNode } from "./nodes/EndNode";
import { StandardNode } from "./nodes/StandardNode";
import { StartAloneNode } from "./nodes/StartAloneNode";
import { StartNode } from "./nodes/StartNode";

const nodeTypes = {
  standard: StandardNode,
  start: StartNode,
  startAlone: StartAloneNode,
  end: EndNode,
};
const edgeTypes = {
  add: AddEdge,
  custom: CustomEdge,
};

export const Canvas = () => {
  const containerSx = {
    position: "relative",
    width: "100%",
    height: "100%",
    fontSize: "0.75rem",
    "& .react-flow__panel": {
      margin: 0,
    },
    "& .react-flow__node-start": {
      alignItems: "flex-end",
    },
    "& .react-flow__node-end": {
      justifyContent: "flex-start",
    },
    "& .react-flow__node": {
      width: `${DEFAULT_NODE_WIDTH}px`,
      height: `${DEFAULT_NODE_HEIGHT}px`,

      display: "flex",
      justifyContent: "center",
      // REACTFLOW - resetting the z-index so it doesn't interfere with elements added on top (e.g.: ContextMenu)
      // https://stackoverflow.com/questions/72483368/how-to-make-edges-appear-above-nodes-in-react-flow
      zIndex: `-1 !important`,
    },
    "& .react-flow__node-startAlone": {
      height: "auto",
    },
    "& .react-flow__edges": {
      // REACTFLOW - resetting the z-index so it doesn't interfere with elements added on top (e.g.: ContextMenu)
      // https://stackoverflow.com/questions/72483368/how-to-make-edges-appear-above-nodes-in-react-flow
      zIndex: `-1 !important`,
    },
    "& .react-flow__edge-add": {
      stroke: "rasaNeutral.500",
    },
    "& .react-flow__edge-path": {
      stroke: "rasaNeutral.500",
    },
    "& .react-flow__handle": {
      visibility: "hidden",
    },
  };

  const { flowName, nodes, edges, handleInit, handleNodesChange } =
    useCanvasContext();

  return (
    <Box css={containerSx} data-testid="canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onInit={handleInit}
        onNodesChange={handleNodesChange}
        nodesDraggable={false}
        nodesConnectable={false}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        proOptions={{ hideAttribution: true }}
        deleteKeyCode={null}
        selectionKeyCode={null}
        multiSelectionKeyCode={null}
        minZoom={0.1}
        maxZoom={5}
      />
      <BottomControls />
      <InspectorViewHeader title="Current flow: " text={flowName} sticky />
    </Box>
  );
};
