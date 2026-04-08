import { Flex, IconButton } from "@chakra-ui/react";
import {
  Brain,
  ClockRotateLeft,
  Columns3,
  Icon,
  Share,
  Sliders,
} from "../Icon";
import { PopoverMenu } from "../PopoverMenu";
import { PopoverMenuItem } from "../PopoverMenuItem";
import { useIsLargeScreen } from "../hooks/useIsLargeScreen";
import { useInspectorStore } from "../store";
import { setInspectorView } from "../store/actions";
import { InspectorView } from "../types/inspector";

export const InspectorViewPopover = () => {
  const inspectorView = useInspectorStore((s) => s.inspectorView);
  const isLargeScreen = useIsLargeScreen();

  return (
    <Flex gap="0.25rem" alignItems="center">
      <PopoverMenu
        trigger={
          <IconButton
            aria-label="Show inspector view options"
            data-testid="show-button"
            variant="solid"
            colorPalette="light"
            size="sm"
          >
            <Icon icon={Sliders} />
          </IconButton>
        }
        header="Show:"
      >
        {isLargeScreen && (
          <PopoverMenuItem
            icon={Columns3}
            label="All"
            isSelected={inspectorView === InspectorView.All}
            onClick={() => {
              setInspectorView(InspectorView.All);
            }}
            testId="view-menu-all"
          />
        )}
        <PopoverMenuItem
          icon={Share}
          label="Active flow"
          isSelected={inspectorView === InspectorView.ActiveFlow}
          onClick={() => {
            setInspectorView(InspectorView.ActiveFlow);
          }}
          testId="view-menu-active-flow"
        />
        <PopoverMenuItem
          icon={ClockRotateLeft}
          label="Flow history"
          isSelected={inspectorView === InspectorView.History}
          onClick={() => {
            setInspectorView(InspectorView.History);
          }}
          testId="view-menu-flow-history"
        />
        <PopoverMenuItem
          icon={Brain}
          label="Memory"
          isSelected={inspectorView === InspectorView.Memory}
          onClick={() => {
            setInspectorView(InspectorView.Memory);
          }}
          testId="view-menu-memory"
        />
      </PopoverMenu>
    </Flex>
  );
};
