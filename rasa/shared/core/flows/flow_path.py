from dataclasses import dataclass, field
from typing import List, Optional, Set

import structlog

NODE_KEY_SEPARATOR = " | "

structlogger = structlog.get_logger()


@dataclass
class PathNode:
    """Representation of a path step."""

    step_id: str
    """Step ID"""

    flow: str
    """Flow name"""

    lines: Optional[str] = None
    """Line numbers range from the original flow .yaml file"""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathNode):
            return False

        return self.flow == other.flow and self.step_id == other.step_id

    def __hash__(self) -> int:
        return hash((self.flow, self.step_id))


@dataclass
class FlowPath:
    """Representation of a path through a flow.

    Attributes:
        flow (str): The name of the flow.
        nodes (List[PathNode]): A list of nodes that constitute the path.
        test_name (str): Name of the test from which it was extracted.
        test_passing (bool): Test status: True if 'passed'.
    """

    flow: str
    nodes: List[PathNode] = field(default_factory=list)

    def are_paths_matching(self, other_path: "FlowPath") -> bool:
        """Compares this FlowPath to another to determine if they are identical."""
        if len(self.nodes) != len(other_path.nodes):
            return False
        return all(
            node == other_node for node, other_node in zip(self.nodes, other_path.nodes)
        )


@dataclass
class FlowPathsList:
    """Representing a list of all available paths through a flow.

    Attributes:
        flow (str): The name of the flow.
        paths (List[FlowPath]): All paths of that flow.
    """

    flow: str
    paths: List[FlowPath] = field(default=list)

    def get_unique_nodes(self) -> Set[PathNode]:
        """Returns the unique nodes of all flow paths."""
        nodes = set()

        for path in self.paths:
            for node in path.nodes:
                nodes.add(node)

        return nodes

    def get_number_of_unique_nodes(self) -> int:
        return len(self.get_unique_nodes())

    def is_path_part_of_list(self, flow_path: FlowPath) -> bool:
        """Checks if the FlowPath exists in a list of FlowPaths."""
        return any(flow_path.are_paths_matching(path) for path in self.paths)
