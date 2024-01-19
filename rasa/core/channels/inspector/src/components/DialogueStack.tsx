import {
  Box,
  FlexProps,
  Heading,
  Table,
  Thead,
  Tbody,
  Tr,
  Th,
  Td,
  Text,
  useColorModeValue,
  Flex,
  Tooltip,
} from "@chakra-ui/react";
import { useOurTheme } from "../theme";
import { SelectedStack, Stack } from "../types";
import { shouldShowTooltip } from "../helpers/utils";

interface Props extends FlexProps {
  stack: Stack[];
  active?: Stack;
  onItemClick?: (selectedStack: SelectedStack) => void;
}

function StackRow({
  stack,
  highlighted,
  selectable,
  onItemClick,
}: {
  stack: Stack;
  highlighted?: boolean;
  selectable?: boolean;
  onItemClick?: (selectedStack: SelectedStack) => void;
}) {
  const { rasaFontSizes } = useOurTheme();

  const idSx = {
    fontSize: rasaFontSizes.xs,
    textTransform: "uppercase",
  };

  // use pointy hand cursor when hovering over a row
  const clickableTrSx = {
    _hover: {
      cursor: "pointer",
    },
  };

  const highlightedTrSx = {
    td: {
      bg: useColorModeValue("warning.50", "warning.50"),
    },
    _last: {
      td: {
        border: "none",
      },
    },
  };

  return (
    <Tr
      sx={highlighted ? highlightedTrSx : selectable ? clickableTrSx : undefined}
      onClick={() => onItemClick?.({stack: stack, isUserSelected: true})}
    >
      <Td sx={idSx}>
        <Text noOfLines={1}>{stack.frame_id}</Text>
      </Td>
      <Td>
        {shouldShowTooltip(stack.flow_id) ? (
          <Tooltip label={stack.flow_id} hasArrow>
            <Text noOfLines={1}>{stack.flow_id}</Text>
          </Tooltip>
        ) : (
          <Text noOfLines={1}>{stack.flow_id}</Text>
        )}
      </Td>
      <Td>
        {shouldShowTooltip(stack.step_id) ? (
          <Tooltip label={stack.step_id} hasArrow>
            <Text noOfLines={1}>{stack.step_id}</Text>
          </Tooltip>
        ) : (
          <Text noOfLines={1}>{stack.step_id}</Text>
        )}
      </Td>
    </Tr>
  );
}

export const DialogueStack = ({
  sx,
  stack,
  active,
  onItemClick,
  ...props
}: Props) => {
  const { rasaSpace } = useOurTheme();

  const containerSx = {
    ...sx,
    pr: 0,
    pb: 0,
    flexDirection: "column",
  };
  const overflowBox = {
    height: "100%",
    overflow: "auto",
    pr: rasaSpace[1],
    pb: rasaSpace[0.5],
  };

  return (
    <Flex sx={containerSx} {...props}>
      <Flex>
        <Heading size="lg" mb={rasaSpace[0.5]}>
          Stack
        </Heading>
        <Text ml={rasaSpace[0.25]}>({stack.length} frames)</Text>
      </Flex>
      <Box sx={overflowBox}>
        <Table width="100%" layout="fixed">
          <Thead>
            <Tr>
              <Th width="20%">ID</Th>
              <Th>Flow</Th>
              <Th width="30%">Step ID</Th>
            </Tr>
          </Thead>
          <Tbody>
            {stack.length > 0 &&
              [...stack]
                .reverse()
                .map((stack) => (
                  <StackRow
                    stack={stack}
                    highlighted={stack.frame_id == active?.frame_id}
                    onItemClick={onItemClick}
                    selectable={true}
                  />
                ))}
            {stack.length === 0 && (
              <StackRow stack={{ frame_id: "-", flow_id: "-", step_id: "-" }} />
            )}
          </Tbody>
        </Table>
      </Box>
    </Flex>
  );
};
