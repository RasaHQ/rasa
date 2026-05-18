import { Box, Button, Flex, IconButton } from "@chakra-ui/react";
import { useStore } from "reactflow";
import { Expand, Icon, Minus, Plus } from "../Icon";
import { useCanvasContext } from "../CanvasContext";
import { Tooltip } from "../Tooltip";

export const BottomControls = () => {
  const {
    handleZoomInClick,
    handleZoomOutClick,
    handleZoomTo100,
    handleFitToCanvasClick,
  } = useCanvasContext();
  const zoom = useStore((store) => store.transform[2]);
  const percentage = new Intl.NumberFormat("en-US", { style: "percent" });

  const containerSx = {
    position: "absolute",
    bottom: "4",
    left: "8",
    height: "auto",
  };

  const buttonProps = {
    colorPalette: "gray",
    variant: "subtle" as const,
  };

  return (
    <Flex css={containerSx}>
      <Box // TODO: reuse shared component with /try-assistant/AsjCopilotButton.tsx (i.e. ButtonGroup)
        zIndex={1}
        boxShadow="tooltip"
        borderRadius="lg"
        display="flex"
      >
        <Tooltip content="Fit to canvas" showArrow>
          <IconButton
            {...buttonProps}
            aria-label="Fit to canvas"
            onClick={handleFitToCanvasClick}
          >
            <Icon icon={Expand} />
          </IconButton>
        </Tooltip>
        <Tooltip content="Zoom in" showArrow>
          <IconButton
            {...buttonProps}
            aria-label="Zoom in"
            onClick={handleZoomInClick}
          >
            <Icon icon={Plus} />
          </IconButton>
        </Tooltip>
        <Tooltip content="Zoom out" showArrow>
          <IconButton
            {...buttonProps}
            aria-label="Zoom out"
            onClick={handleZoomOutClick}
          >
            <Icon icon={Minus} />
          </IconButton>
        </Tooltip>
        <Tooltip content="Zoom to 100%" showArrow>
          <Button
            {...buttonProps}
            pr="2"
            pl="1"
            aria-label="Zoom to 100%"
            onClick={handleZoomTo100}
          >
            {percentage.format(zoom)}
          </Button>
        </Tooltip>
      </Box>
    </Flex>
  );
};
