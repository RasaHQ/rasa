from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow_path import (
    FlowPath,
    PathNode,
    FlowPathsList,
)


@pytest.fixture
def sample_path_nodes() -> List[PathNode]:
    """Provides sample PathNode objects for testing."""
    return [
        PathNode(step_id="Hello", flow="flow1", lines="1-2"),
        PathNode(step_id="World", flow="flow1", lines="3-4"),
    ]


@pytest.fixture
def sample_flow_path(sample_path_nodes: List[PathNode]) -> FlowPath:
    """Provides a sample FlowPath object for testing."""
    return FlowPath(flow="flow1", nodes=sample_path_nodes)


@pytest.fixture
def another_flow_path() -> FlowPath:
    """Provides another sample FlowPath object for testing."""
    return FlowPath(
        flow="flow1",
        nodes=[
            PathNode(step_id="Hello", flow="flow1", lines="1-2"),
            PathNode(step_id="World", flow="flow1", lines="3-4"),
        ],
    )


def test_path_node_equality(sample_path_nodes: List[PathNode]) -> None:
    """Tests equality of PathNode objects.

    Args:
        sample_path_nodes (List[PathNode]): Sample PathNode objects for testing.

    Asserts:
        PathNode objects are compared correctly for equality.
    """
    node1, node2 = sample_path_nodes
    assert node1 == PathNode(step_id="Hello", flow="flow1", lines="1-2")
    assert node2 == PathNode(step_id="World", flow="flow1", lines="3-4")
    assert node1 != PathNode(step_id="Different", flow="flow1", lines="1-2")


def test_flow_path_match(
    sample_flow_path: FlowPath, another_flow_path: FlowPath
) -> None:
    """Tests the matching of FlowPath objects.

    Args:
        sample_flow_path (FlowPath): A sample FlowPath object for testing.
        another_flow_path (FlowPath): Another sample FlowPath object for testing.

    Asserts:
        FlowPath objects are compared correctly for matching.
    """
    assert sample_flow_path.are_paths_matching(another_flow_path)
    different_flow_path = FlowPath(
        flow="flow1", nodes=[PathNode(step_id="Different", flow="flow1", lines="1-2")]
    )
    assert not sample_flow_path.are_paths_matching(different_flow_path)


def test_is_in_list_of_paths(
    sample_flow_path: FlowPath, another_flow_path: FlowPath
) -> None:
    """Tests if a FlowPath exists in a list of FlowPaths.

    Args:
        sample_flow_path (FlowPath): A sample FlowPath object for testing.
        another_flow_path (FlowPath): Another sample FlowPath object for testing.

    Asserts:
        FlowPath is correctly identified in the list of FlowPaths.
    """
    paths_list = FlowPathsList("flow1", [another_flow_path])
    assert paths_list.is_path_part_of_list(sample_flow_path)
    different_flow_path = FlowPath(
        flow="flow1", nodes=[PathNode(step_id="Different", flow="flow1", lines="1-2")]
    )
    paths_list = FlowPathsList("flow1", [different_flow_path])
    assert not paths_list.is_path_part_of_list(sample_flow_path)


def test_extract_paths_from_flows(tmp_path: Path) -> None:
    """Tests the extraction of paths from flow definitions and saving to a YAML file."""
    mock_flow = MagicMock()
    mock_flow.id = "flow1"
    mock_flow.is_rasa_default_flow = False
    mock_flow.extract_all_paths.return_value = FlowPathsList(
        "flow1",
        [
            FlowPath(
                flow="flow1",
                nodes=[
                    PathNode(step_id="Hello", flow="flow1", lines="1-2"),
                    PathNode(step_id="World", flow="flow1", lines="3-4"),
                ],
            )
        ],
    )
    flows = FlowsList([mock_flow])

    extracted_paths = flows.extract_flow_paths()

    assert extracted_paths["flow1"].flow == "flow1"
    assert len(extracted_paths["flow1"].paths) == 1


def test_get_unique_nodes() -> None:
    flow_path_list = FlowPathsList(
        "flow_a",
        paths=[
            FlowPath(
                "flow_a",
                [
                    PathNode("step_1", "flow_a"),
                    PathNode("step_2", "flow_a"),
                ],
            ),
            FlowPath(
                "flow_a", [PathNode("step_1", "flow_a"), PathNode("step_4", "flow_a")]
            ),
            FlowPath(
                "flow_a",
                [
                    PathNode("step_1", "flow_a"),
                ],
            ),
        ],
    )

    unique_nodes = flow_path_list.get_unique_nodes()

    assert PathNode("step_1", "flow_a") in unique_nodes
    assert PathNode("step_2", "flow_a") in unique_nodes
    assert PathNode("step_4", "flow_a") in unique_nodes


def test_get_number_of_unique_nodes() -> None:
    flow_path_list = FlowPathsList(
        "flow_a",
        paths=[
            FlowPath(
                "flow_a",
                [
                    PathNode("step_1", "flow_a"),
                    PathNode("step_2", "flow_a"),
                ],
            ),
            FlowPath(
                "flow_a", [PathNode("step_1", "flow_a"), PathNode("step_3", "flow_a")]
            ),
        ],
    )

    count = flow_path_list.get_number_of_unique_nodes()

    assert count == 3
