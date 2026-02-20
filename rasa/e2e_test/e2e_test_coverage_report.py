from typing import Any, Dict, List, Optional, Set

import pandas as pd
import structlog

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    KnowledgeAnswerCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow_path import FlowPath, FlowPathsList, PathNode

# Report column names
FLOW_NAME_COL_NAME = "Flow Name"
NUM_STEPS_COL_NAME = "Num Steps"
MISSING_STEPS_COL_NAME = "Missing Steps"
LINE_NUMBERS_COL_NAME = "Line Numbers for Missing Steps"
COVERAGE_COL_NAME = "Coverage"

FLOWS_KEY = "flows"
NUMBER_OF_STEPS_KEY = "number_of_steps"
NUMBER_OF_UNTESTED_STEPS_KEY = "number_of_untested_steps"
UNTESTED_LINES_KEY = "untested_lines"

SUPPORTED_HISTOGRAM_COMMANDS = [
    KnowledgeAnswerCommand.command(),
    StartFlowCommand.command(),
    SetSlotCommand.command(),
    ClarifyCommand.command(),
    CancelFlowCommand.command(),
    ChitChatAnswerCommand.command(),
    SkipQuestionCommand.command(),
]

structlogger = structlog.get_logger()


def create_coverage_report(
    flows: FlowsList,
    test_results: List[TestResult],
) -> Optional[pd.DataFrame]:
    """Generates a coverage report.

    This function extracts paths from predefined flows, loads tested paths based
    on the provided e2e test results, and compares the unique nodes of all flow paths
    of one flow to obtain untested nodes. It then generates
    a report that highlights areas of the flows that are not adequately tested.

    Coverage is calculated per flow based on the nodes' actual flow ID, not the
    parent flow that triggered the path extraction. This ensures that:
    - Called/linked flows have their coverage tracked separately
    - Line numbers in the report match the actual flow files
    - Low coverage in a called flow doesn't pollute parent flows' coverage

    Args:
        flows: List of flows.
        test_results: List of e2e test results.

    Returns:
        The coverage report as dataframe.
    """
    if not test_results or flows.user_flows.is_empty():
        return _empty_dataframe()

    # Step 1: Get testable flow paths
    flow_to_testable_paths = flows.extract_flow_paths()

    # Step 2: Get tested flow paths from e2e tests run
    tested_flow_paths = _extract_tested_flow_paths(test_results)
    # No tested flow paths exists, cannot create coverage report
    if not tested_flow_paths:
        return _empty_dataframe()

    # Step 3: Group flow paths by flow
    flow_to_tested_paths = _group_flow_paths_by_flow(tested_flow_paths)

    # Step 4: Get all testable nodes grouped by their actual flow ID
    # (nodes may belong to different flows than the path's starting flow)
    all_testable_nodes = _collect_all_testable_nodes(flow_to_testable_paths)
    all_tested_nodes = _collect_all_tested_nodes(flow_to_tested_paths)

    # Step 5: Get the unvisited nodes and number of unique nodes per flow
    # based on each node's actual flow ID
    unvisited_nodes_per_flow = _get_unvisited_nodes_per_flow(
        all_testable_nodes, all_tested_nodes
    )
    number_of_nodes_per_flow = {
        flow: len(nodes) for flow, nodes in all_testable_nodes.items()
    }

    # Step 6: Produce the report
    coverage_report_data = _create_coverage_report_data(
        flows,
        number_of_nodes_per_flow,
        unvisited_nodes_per_flow,
    )
    return _create_data_frame(coverage_report_data)


def _collect_all_testable_nodes(
    flow_to_testable_paths: Dict[str, FlowPathsList],
) -> Dict[str, Set[PathNode]]:
    """Collects all testable nodes grouped by their actual flow ID.

    Nodes are grouped by the node's own flow attribute, not the flow that
    triggered the path extraction. This handles called/linked flows correctly.

    Args:
        flow_to_testable_paths: Testable paths per flow.

    Returns:
        Dict mapping flow ID to set of testable nodes belonging to that flow.
    """
    nodes_per_flow: Dict[str, Set[PathNode]] = {}

    for flow_paths in flow_to_testable_paths.values():
        for path in flow_paths.paths:
            for node in path.nodes:
                if node.flow not in nodes_per_flow:
                    nodes_per_flow[node.flow] = set()
                nodes_per_flow[node.flow].add(node)

    return nodes_per_flow


def _collect_all_tested_nodes(
    flow_to_tested_paths: Dict[str, FlowPathsList],
) -> Dict[str, Set[PathNode]]:
    """Collects all tested nodes grouped by their actual flow ID.

    Nodes are grouped by the node's own flow attribute, not the flow that
    triggered the path extraction. This handles called/linked flows correctly.

    Args:
        flow_to_tested_paths: Tested paths per flow.

    Returns:
        Dict mapping flow ID to set of tested nodes belonging to that flow.
    """
    nodes_per_flow: Dict[str, Set[PathNode]] = {}

    for flow_paths in flow_to_tested_paths.values():
        for path in flow_paths.paths:
            for node in path.nodes:
                if node.flow not in nodes_per_flow:
                    nodes_per_flow[node.flow] = set()
                nodes_per_flow[node.flow].add(node)

    return nodes_per_flow


def _get_unvisited_nodes_per_flow(
    all_testable_nodes: Dict[str, Set[PathNode]],
    all_tested_nodes: Dict[str, Set[PathNode]],
) -> Dict[str, Set[PathNode]]:
    """Returns the unvisited nodes per flow based on each node's actual flow ID.

    Compares testable nodes to tested nodes for each flow. Nodes are grouped
    by their actual flow ID, ensuring that coverage for called/linked flows
    is tracked separately from the parent flow.

    Args:
        all_testable_nodes: Dict mapping flow ID to set of testable nodes.
        all_tested_nodes: Dict mapping flow ID to set of tested nodes.

    Returns:
        Dict mapping flow ID to set of unvisited nodes.
    """
    unvisited_nodes_per_flow: Dict[str, Set[PathNode]] = {}

    for flow, testable_nodes in all_testable_nodes.items():
        if flow in all_tested_nodes:
            # get the difference of testable and tested nodes
            unvisited_nodes = testable_nodes.difference(all_tested_nodes[flow])
        else:
            # the flow was not tested at all
            unvisited_nodes = testable_nodes
        unvisited_nodes_per_flow[flow] = unvisited_nodes

    return unvisited_nodes_per_flow


def _group_flow_paths_by_flow(flow_paths: List[FlowPath]) -> Dict[str, FlowPathsList]:
    """Group the available flow paths by flow id.

    Args:
        flow_paths: The list of all flow paths.

    Returns:
        A dictionary mapping a flow to its paths.
    """
    flow_to_paths = {}

    for flow_path in flow_paths:
        if flow_path.flow not in flow_to_paths:
            flow_to_paths[flow_path.flow] = FlowPathsList(flow_path.flow, [flow_path])
        else:
            flow_to_paths[flow_path.flow].paths.append(flow_path)

    return flow_to_paths


def _create_data_frame(coverage_report_data: Dict[str, Any]) -> pd.DataFrame:
    df = _construct_dataframe(coverage_report_data)
    df = _calculate_coverage(df)
    df = _append_total_row(df)
    df = _reorder_columns(df)
    return df


def _empty_dataframe() -> pd.DataFrame:
    """Generates an empty DataFrame.

    The DataFrame includes all required columns for a coverage report.

    Returns:
        pd.DataFrame: An empty DataFrame with predefined columns.
    """
    return pd.DataFrame(
        columns=[
            FLOW_NAME_COL_NAME,
            COVERAGE_COL_NAME,
            NUM_STEPS_COL_NAME,
            MISSING_STEPS_COL_NAME,
            LINE_NUMBERS_COL_NAME,
        ]
    )


def _create_coverage_report_data(
    flows: FlowsList,
    number_of_nodes_per_flow: Dict[str, int],
    unvisited_nodes_per_flow: Dict[str, Set[PathNode]],
) -> Dict[str, Any]:
    """Creates the data for the coverage report.

    Args:
        flows: All available flows
        number_of_nodes_per_flow: Number of nodes per flow
        unvisited_nodes_per_flow: Unvisited nodes per flow

    Returns:
        A dictionary with processed data needed to construct the DataFrame.
    """
    # Include all flows that have testable nodes (including called/linked flows)
    # Use number_of_nodes_per_flow keys to get all flows with testable nodes
    flow_ids = list(number_of_nodes_per_flow.keys())

    flow_full_names: List[str] = []
    number_of_steps: List[int] = []
    number_of_untested_steps: List[int] = []
    untested_lines: List[List[str]] = []

    for flow in flow_ids:
        nodes = unvisited_nodes_per_flow.get(flow, set())
        lines: List[str] = [node.lines for node in nodes if node.lines]

        if flow_object := flows.flow_by_id(flow):
            flow_full_names.append(flow_object.get_full_name())
        else:
            # Flow might be a pattern or default flow
            flow_full_names.append(flow)
        number_of_steps.append(number_of_nodes_per_flow.get(flow, 0))
        number_of_untested_steps.append(len(nodes))
        untested_lines.append(lines)

    return {
        FLOWS_KEY: flow_full_names,
        NUMBER_OF_STEPS_KEY: number_of_steps,
        NUMBER_OF_UNTESTED_STEPS_KEY: number_of_untested_steps,
        UNTESTED_LINES_KEY: _reformat_untested_lines(untested_lines),
    }


def _reformat_untested_lines(untested_lines: List[List[str]]) -> List[str]:
    """Format lists of lists to list of str.

    Formats nested lists of untested lines into a string format suitable
    for display.

    Args:
        untested_lines: A list of lists containing line information.

    Returns:
        A list of formatted strings representing untested lines.
    """
    return ["[" + ", ".join(sublist) + "]" for sublist in untested_lines]


def _construct_dataframe(report_data: Dict[str, Any]) -> pd.DataFrame:
    """Constructs a DataFrame from the provided data.

    Args:
        report_data: A dictionary containing report data.

    Returns:
        pd.DataFrame: A DataFrame constructed from the report data.
    """
    return pd.DataFrame(
        {
            FLOW_NAME_COL_NAME: report_data[FLOWS_KEY],
            NUM_STEPS_COL_NAME: report_data[NUMBER_OF_STEPS_KEY],
            MISSING_STEPS_COL_NAME: report_data[NUMBER_OF_UNTESTED_STEPS_KEY],
            LINE_NUMBERS_COL_NAME: report_data[UNTESTED_LINES_KEY],
        }
    )


def _calculate_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the coverage percentage and updates the DataFrame.

    Args:
        df: The DataFrame to update with coverage data.

    Returns:
        The updated DataFrame with coverage percentages.
    """
    df[COVERAGE_COL_NAME] = (
        (df[NUM_STEPS_COL_NAME] - df[MISSING_STEPS_COL_NAME]) / df[NUM_STEPS_COL_NAME]
    ) * 100

    # Set the float format for displaying coverage percentages
    pd.options.display.float_format = "{:.2f}%".format

    return df


def _append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    """Appends a total summary row to the DataFrame.

    Args:
        df: The DataFrame to which the total row will be appended.

    Returns:
        The updated DataFrame with an appended total row.
    """
    total_data = {
        FLOW_NAME_COL_NAME: "Total",
        NUM_STEPS_COL_NAME: df[NUM_STEPS_COL_NAME].sum(),
        MISSING_STEPS_COL_NAME: df[MISSING_STEPS_COL_NAME].sum(),
        LINE_NUMBERS_COL_NAME: "",
        COVERAGE_COL_NAME: _calculate_total_coverage(df),
    }
    # Append the total row using `.loc`
    df.loc[len(df)] = total_data

    return df


def _calculate_total_coverage(df: pd.DataFrame) -> str:
    """Calculates the total coverage percentage for the DataFrame.

    Args:
        df: The DataFrame for which total coverage is calculated.

    Returns:
        The calculated total coverage percentage formatted as a string.
    """
    total_steps = df[NUM_STEPS_COL_NAME].sum()
    missing_steps = df[MISSING_STEPS_COL_NAME].sum()
    return (
        pd.NA
        if total_steps == 0
        else f"{((total_steps - missing_steps) / total_steps) * 100:.2f}%"
    )


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorders the columns of the DataFrame to prioritize specific columns.

    Args:
        df: The DataFrame whose columns are to be reordered.

    Returns:
        The DataFrame with reordered columns.
    """
    columns = [FLOW_NAME_COL_NAME, COVERAGE_COL_NAME] + [
        col for col in df.columns if col not in [FLOW_NAME_COL_NAME, COVERAGE_COL_NAME]
    ]
    return df[columns]


def _extract_tested_flow_paths(test_results: List[TestResult]) -> List[FlowPath]:
    """Extract the flow paths of the test results.

    Args:
        test_results: List of test results.

    Returns:
        List[FlowPath]: A list of FlowPaths.
    """
    flatten_paths = []

    for test_result in test_results:
        if test_result.tested_paths:
            for path in test_result.tested_paths:
                flatten_paths.append(path)

    return flatten_paths


def extract_tested_commands(test_results: List[TestResult]) -> Dict[str, int]:
    """Extract tested commands from the test results.

    Args:
        test_results: List of test results.

    Returns:
        Dict[str, int]: A dictionary of commands and their counts.
    """
    command_histogram_data = {}
    for command in SUPPORTED_HISTOGRAM_COMMANDS:
        command_histogram_data[command] = 0

    for test_result in test_results:
        if test_result.tested_commands:
            for flow, commands_dict in test_result.tested_commands.items():
                for command, count in commands_dict.items():
                    if command in command_histogram_data:
                        command_histogram_data[command] += count

    return dict(command_histogram_data)
