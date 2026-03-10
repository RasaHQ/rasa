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
    bottom: "1rem",
    left: "2rem",
    height: "auto",
  };

  const buttonProps = {
    colorPalette: "dark",
    variant: "subtle" as const,
  };

  return (
    <Flex css={containerSx}>
      <Box // TODO: reuse shared component with /try-assistant/AsjCopilotButton.tsx (i.e. ButtonGroup)
        zIndex={1}
        bg="rasaNeutral.50"
        boxShadow="0px 2px 8px 0px #00000026"
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
            pr="0.5rem"
            pl="0.25rem"
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
