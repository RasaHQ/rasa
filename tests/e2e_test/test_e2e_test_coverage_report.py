from typing import List, Dict
from unittest.mock import MagicMock

import pandas as pd
import pytest

from rasa.dialogue_understanding.commands import (
    SetSlotCommand,
    StartFlowCommand,
    KnowledgeAnswerCommand,
    ChitChatAnswerCommand,
    CancelFlowCommand,
    HumanHandoffCommand,
    NoopCommand,
    SkipQuestionCommand,
    ClarifyCommand,
)
from rasa.e2e_test.e2e_test_coverage_report import (
    _empty_dataframe,
    FLOW_NAME_COL_NAME,
    COVERAGE_COL_NAME,
    NUM_STEPS_COL_NAME,
    MISSING_STEPS_COL_NAME,
    LINE_NUMBERS_COL_NAME,
    _construct_dataframe,
    _calculate_coverage,
    _append_total_row,
    _reorder_columns,
    create_coverage_report,
    FLOWS_KEY,
    NUMBER_OF_STEPS_KEY,
    NUMBER_OF_UNTESTED_STEPS_KEY,
    UNTESTED_LINES_KEY,
    _extract_tested_flow_paths,
    _create_coverage_report_data,
    _get_unvisited_nodes_per_flow,
    _group_flow_paths_by_flow,
    extract_tested_commands,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow_path import FlowPath, PathNode, FlowPathsList
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader


@pytest.fixture
def sample_test_results(sample_tested_flow_paths: List[FlowPath]) -> List[TestResult]:
    test_result = TestResult(
        MagicMock(),
        pass_status=True,
        difference=[],
        tested_paths=sample_tested_flow_paths,
    )
    return [test_result]


@pytest.fixture
def sample_flows() -> FlowsList:
    definition = """flows:
      flow1:
        name: flow1
        description: flow1
        steps:
        - collect: step1
        - collect: step2

      flow2:
        name: flow2
        description: flow2
        steps:
          - collect: step1
          - collect: step2
          - collect: step3
    """
    return YAMLFlowsReader.read_from_string(definition)


@pytest.fixture
def sample_tested_flow_paths() -> List[FlowPath]:
    """Provides sample tested flow paths for testing."""
    path1 = FlowPath(
        flow="flow1",
        nodes=[
            PathNode(step_id="0_collect_step1", flow="flow1"),
        ],
    )
    path2 = FlowPath(
        flow="flow2",
        nodes=[
            PathNode(step_id="0_collect_step1", flow="flow2"),
            PathNode(step_id="1_collect_step2", flow="flow2"),
            PathNode(step_id="2_collect_step3", flow="flow2"),
        ],
    )
    return [path1, path2]


def test_empty_dataframe() -> None:
    """Tests the creation of an empty DataFrame.

    Asserts:
        The DataFrame is empty and has the correct columns.
    """
    df = _empty_dataframe()
    assert df.empty
    assert set(df.columns) == {
        FLOW_NAME_COL_NAME,
        COVERAGE_COL_NAME,
        NUM_STEPS_COL_NAME,
        MISSING_STEPS_COL_NAME,
        LINE_NUMBERS_COL_NAME,
    }


def test_extract_tested_flow_paths(
    sample_test_results: List[TestResult],
) -> None:
    flow_paths = _extract_tested_flow_paths(sample_test_results)
    assert isinstance(flow_paths, List)
    assert len(flow_paths) == 2
    assert isinstance(flow_paths[0], FlowPath)


def test_construct_dataframe() -> None:
    """Tests the construction of a DataFrame from flow data.

    Asserts:
        The DataFrame is constructed correctly with the right columns and data.
    """
    flow_data = {
        FLOWS_KEY: ["flow1", "flow2"],
        NUMBER_OF_STEPS_KEY: [2, 1],
        NUMBER_OF_UNTESTED_STEPS_KEY: [2, 1],
        UNTESTED_LINES_KEY: ["[1-2]", "[3-4]"],
    }
    df = _construct_dataframe(flow_data)
    assert list(df.columns) == [
        FLOW_NAME_COL_NAME,
        NUM_STEPS_COL_NAME,
        MISSING_STEPS_COL_NAME,
        LINE_NUMBERS_COL_NAME,
    ]
    assert df.shape == (2, 4)
    assert df.at[0, FLOW_NAME_COL_NAME] == "flow1"
    assert df.at[1, FLOW_NAME_COL_NAME] == "flow2"


def test_calculate_coverage() -> None:
    """Tests the calculation of coverage percentages.

    Asserts:
        The calculated coverage percentages are correct.
    """
    data = {
        FLOW_NAME_COL_NAME: ["flow1", "flow2"],
        NUM_STEPS_COL_NAME: [2, 1],
        MISSING_STEPS_COL_NAME: [1, 1],
    }
    df = pd.DataFrame(data)
    df = _calculate_coverage(df)
    assert df.at[0, COVERAGE_COL_NAME] == 50.0
    assert df.at[1, COVERAGE_COL_NAME] == 0.0


def test_append_total_row() -> None:
    """Tests appending a total row to the DataFrame.

    Asserts:
        The total row is appended correctly with the right data.
    """
    data = {
        FLOW_NAME_COL_NAME: ["flow1", "flow2"],
        NUM_STEPS_COL_NAME: [2, 1],
        MISSING_STEPS_COL_NAME: [1, 1],
        LINE_NUMBERS_COL_NAME: ["[1-2]", "[3-4]"],
        COVERAGE_COL_NAME: ["50.00%", "0.00%"],
    }
    df = pd.DataFrame(data)
    df = _append_total_row(df)
    assert df.iloc[-1][FLOW_NAME_COL_NAME] == "Total"
    assert df.iloc[-1][NUM_STEPS_COL_NAME] == 3
    assert df.iloc[-1][MISSING_STEPS_COL_NAME] == 2
    assert df.iloc[-1][COVERAGE_COL_NAME] == "33.33%"


def test_reorder_columns() -> None:
    """Tests reordering the columns of the DataFrame.

    Asserts:
        The columns are reordered correctly.
    """
    data = {
        FLOW_NAME_COL_NAME: ["flow1"],
        COVERAGE_COL_NAME: ["50.00%"],
        NUM_STEPS_COL_NAME: [2],
        MISSING_STEPS_COL_NAME: [1],
        LINE_NUMBERS_COL_NAME: ["[1-2]"],
    }
    df = pd.DataFrame(data)
    df = _reorder_columns(df)
    assert list(df.columns) == [
        FLOW_NAME_COL_NAME,
        COVERAGE_COL_NAME,
        NUM_STEPS_COL_NAME,
        MISSING_STEPS_COL_NAME,
        LINE_NUMBERS_COL_NAME,
    ]


def test_create_coverage_report_data() -> None:
    """Tests generating the coverage report data."""
    data = _create_coverage_report_data(
        flow_names=["flow_a", "flow_b"],
        number_of_nodes_per_flow={"flow_a": 5, "flow_b": 2},
        unvisited_nodes_per_flow={
            "flow_a": {PathNode(step_id="step_1", flow="flow_a", lines="2-3")},
            "flow_b": set(),
        },
    )

    assert data[FLOWS_KEY] == ["flow_a", "flow_b"]
    assert data[NUMBER_OF_STEPS_KEY] == [5, 2]
    assert data[NUMBER_OF_UNTESTED_STEPS_KEY] == [1, 0]
    assert data[UNTESTED_LINES_KEY] == ["[2-3]", "[]"]


def test_get_unvisited_nodes_per_flow():
    flow_to_testable_paths = {
        "flow_a": FlowPathsList(
            "flow_a",
            paths=[
                FlowPath(
                    "flow_a",
                    [
                        PathNode("step_1", "flow_a"),
                        PathNode("step_2", "flow_a"),
                        PathNode("step_3", "flow_a"),
                    ],
                ),
                FlowPath(
                    "flow_a",
                    [
                        PathNode("step_1", "flow_a"),
                        PathNode("step_3", "flow_a"),
                        PathNode("step_4", "flow_a"),
                    ],
                ),
                FlowPath(
                    "flow_a",
                    [
                        PathNode("step_1", "flow_a"),
                        PathNode("step_2", "flow_a"),
                    ],
                ),
            ],
        ),
        "flow_b": FlowPathsList(
            "flow_b",
            paths=[
                FlowPath(
                    "flow_b",
                    [
                        PathNode("step_1", "flow_b"),
                        PathNode("step_2", "flow_b"),
                        PathNode("step_3", "flow_b"),
                    ],
                ),
            ],
        ),
    }
    flow_to_tested_paths = {
        "flow_a": FlowPathsList(
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
                    "flow_a",
                    [PathNode("step_1", "flow_a"), PathNode("step_4", "flow_a")],
                ),
                FlowPath(
                    "flow_a",
                    [
                        PathNode("step_1", "flow_a"),
                    ],
                ),
            ],
        )
    }

    unvisited_nodes_per_flow = _get_unvisited_nodes_per_flow(
        flow_to_testable_paths, flow_to_tested_paths
    )

    assert unvisited_nodes_per_flow["flow_a"] == {PathNode("step_3", "flow_a")}
    assert unvisited_nodes_per_flow["flow_b"] == {
        PathNode("step_1", "flow_b"),
        PathNode("step_2", "flow_b"),
        PathNode("step_3", "flow_b"),
    }


def test_group_flow_paths_by_flow():
    flow_paths = [
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
            "flow_c",
            [
                PathNode("step_1", "flow_c"),
            ],
        ),
        FlowPath(
            "flow_b",
            [
                PathNode("step_1", "flow_b"),
                PathNode("step_2", "flow_b"),
                PathNode("step_3", "flow_b"),
            ],
        ),
    ]

    flow_to_paths = _group_flow_paths_by_flow(flow_paths)

    assert "flow_a" in flow_to_paths
    assert len(flow_to_paths["flow_a"].paths) == 2
    assert "flow_b" in flow_to_paths
    assert len(flow_to_paths["flow_b"].paths) == 1
    assert "flow_c" in flow_to_paths
    assert len(flow_to_paths["flow_c"].paths) == 1


def test_create_coverage_report(
    sample_test_results: List[TestResult],
    sample_flows: FlowsList,
) -> None:
    """Tests the create_coverage_report function.

    Args:
        sample_test_results (List[TestResult]): Mock for the test results
        sample_flows (List[Flow]): Sample flows for testing.

    Asserts:
        The function completes without errors and the coverage report is generated.
        The content of the report DataFrame is correct.
    """
    df = create_coverage_report(sample_flows, sample_test_results)

    assert not df.empty
    assert len(df) == 3
    assert df[FLOW_NAME_COL_NAME][0] == "flow1"
    assert df[FLOW_NAME_COL_NAME][1] == "flow2"
    assert df[FLOW_NAME_COL_NAME][2] == "Total"
    assert df[NUM_STEPS_COL_NAME][0] == 2
    assert df[NUM_STEPS_COL_NAME][1] == 3
    assert df[NUM_STEPS_COL_NAME][2] == 5
    assert df[MISSING_STEPS_COL_NAME][0] == 1
    assert df[MISSING_STEPS_COL_NAME][1] == 0
    assert df[MISSING_STEPS_COL_NAME][2] == 1
    assert df[COVERAGE_COL_NAME][0] == 50.0
    assert df[COVERAGE_COL_NAME][1] == 100.0
    assert df[COVERAGE_COL_NAME][2] == "80.00%"
    assert df[LINE_NUMBERS_COL_NAME][0] == "[7-8]"
    assert df[LINE_NUMBERS_COL_NAME][1] == "[]"
    assert df[LINE_NUMBERS_COL_NAME][2] == ""


@pytest.mark.parametrize(
    "test_results, expected_output",
    [
        # Normal Case: Different commands and counts
        (
            [
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow1": {
                            SetSlotCommand.command(): 2,
                            StartFlowCommand.command(): 3,
                        },
                        "flow2": {KnowledgeAnswerCommand.command(): 1},
                    },
                ),
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow3": {
                            SetSlotCommand.command(): 1,
                            ChitChatAnswerCommand.command(): 4,
                        },
                    },
                ),
            ],
            {
                SetSlotCommand.command(): 3,
                StartFlowCommand.command(): 3,
                KnowledgeAnswerCommand.command(): 1,
                ChitChatAnswerCommand.command(): 4,
                CancelFlowCommand.command(): 0,
                HumanHandoffCommand.command(): 0,
                SkipQuestionCommand.command(): 0,
                ClarifyCommand.command(): 0,
            },
        ),
        # Empty Case: Empty list of test results
        (
            [],
            {
                SetSlotCommand.command(): 0,
                StartFlowCommand.command(): 0,
                CancelFlowCommand.command(): 0,
                KnowledgeAnswerCommand.command(): 0,
                HumanHandoffCommand.command(): 0,
                ChitChatAnswerCommand.command(): 0,
                SkipQuestionCommand.command(): 0,
                ClarifyCommand.command(): 0,
            },
        ),
        # Multiple Test Results: Overlapping commands
        (
            [
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow1": {
                            SetSlotCommand.command(): 2,
                            StartFlowCommand.command(): 3,
                        },
                    },
                ),
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow2": {
                            StartFlowCommand.command(): 1,
                            CancelFlowCommand.command(): 2,
                        },
                    },
                ),
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow3": {
                            SetSlotCommand.command(): 1,
                            HumanHandoffCommand.command(): 5,
                        },
                    },
                ),
            ],
            {
                SetSlotCommand.command(): 3,
                StartFlowCommand.command(): 4,
                CancelFlowCommand.command(): 2,
                HumanHandoffCommand.command(): 5,
                KnowledgeAnswerCommand.command(): 0,
                ChitChatAnswerCommand.command(): 0,
                SkipQuestionCommand.command(): 0,
                ClarifyCommand.command(): 0,
            },
        ),
        # Edge Case: Commands with zero counts and no commands in some flows
        (
            [
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow1": {
                            SetSlotCommand.command(): 0,
                            StartFlowCommand.command(): 3,
                            NoopCommand.command(): 10,
                        },
                    },
                ),
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={},
                ),
                TestResult(
                    test_case=MagicMock(),
                    pass_status=True,
                    difference=[],
                    tested_commands={
                        "flow3": {CancelFlowCommand.command(): 2},
                    },
                ),
            ],
            {
                SetSlotCommand.command(): 0,
                StartFlowCommand.command(): 3,
                CancelFlowCommand.command(): 2,
                KnowledgeAnswerCommand.command(): 0,
                HumanHandoffCommand.command(): 0,
                ChitChatAnswerCommand.command(): 0,
                SkipQuestionCommand.command(): 0,
                ClarifyCommand.command(): 0,
            },
        ),
    ],
)
def test_extract_tested_commands(
    test_results: List[TestResult], expected_output: Dict[str, int]
) -> None:
    assert expected_output == extract_tested_commands(test_results)
