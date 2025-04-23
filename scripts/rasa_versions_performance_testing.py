"""Performance testing of CALM CompactLLMCommandGenerator, against baseline."""

import os
import sys
from typing import Optional

import pytest
import yaml

# TODO: Currently this baseline is manually recorded in Notion,
#       so hardcoding baseline here. In future, once baselines
#       are recorded in Honeycomb (or other experiment tracking tool),
#       then baselines can be fetched dynamically from there.
BASELINE = {
    "gpt-4o-2024-11-20": {
        "accuracy": {
            "test_cases": 0.71704576269348,
            "user_utterances": 0.8702198177951251,
        },
        "f1_score": {
            "macro": 0.8374078379191436,
            "micro": 0.909220643896572,
            "weighted_average": 0.902071028214975,
        },
        "command_metrics": {
            "cancel_flow": {"f1_score": 0.9574468085106383},
            "chitchat": {"f1_score": 0.9105356228768207},
            "clarify": {"f1_score": 0.36257820456860174},
            "human_handoff": {"f1_score": 0.9407114624505928},
            "knowledge": {"f1_score": 0.8235847350334599},
            "set_slot": {"f1_score": 0.9584818355565435},
            "start_flow": {"f1_score": 0.9085161964373478},
        },
    },
    "anthropic.claude-3-5-sonnet-20240620-v1": {
        "accuracy": {
            "test_cases": 0.5808,
            "user_utterances": 0.7396,
        },
        "f1_score": {"macro": 0.8041, "micro": 0.8636, "weighted_average": 0.8521},
        "command_metrics": {
            "cancel_flow": {"f1_score": 0.9090909090909091},
            "chitchat": {"f1_score": 0.7999999999999999},
            "clarify": {"f1_score": 0.33333333333333337},
            "human_handoff": {"f1_score": 1},
            "knowledge": {"f1_score": 0.6956521739130435},
            "set_slot": {"f1_score": 0.8286852589641432},
            "start_flow": {"f1_score": 0.9072164948453608},
        },
    },
}


def get_dut_test_results_from_results_file(
    results_file_path: str,
) -> dict[str, dict[str, float] | dict[str, dict[str, float] | dict[str, int]]]:
    """Get results of current DUT tests run, from provided results file, for comparison with baseline.

    Args:
        results_file_path (str): Path to DUT test results file.

    Raises:
        sys.exit: Exit if results file not found or inaccessible.

    Returns:
        dict[str, dict[str, float] | dict[str, dict[str, float] | dict[str, int]]]: Current DUT test run's results.
    """
    try:
        with open(
            results_file_path,
            encoding="utf-8",
        ) as dut_results:
            return yaml.safe_load(dut_results)
    except FileNotFoundError as e:
        raise sys.exit(
            f"DUT baseline results file: {results_file_path} not found"
        ) from e


results = get_dut_test_results_from_results_file(
    os.getenv(
        "DUT_RESULTS_FILE_PATH",
        "current-test-results.yml",
    )
)


@pytest.fixture(scope="session")
def model() -> Optional[str]:
    """Determine model used for DUT tests.

    Raises:
        sys.exit: Exit if no model specified.

    Returns:
        Optional[str]: model name
    """
    try:
        return os.environ["LLM_MODEL_NAME"]
    except KeyError as e:
        raise sys.exit(
            "Model used for test not specified. Please specify 'gpt-4o-2024-11-20' or 'anthropic.claude-3-5-sonnet-20240620-v1'"
        ) from e


@pytest.mark.parametrize("accuracy_metric", {"test_cases", "user_utterances"})
def test_accuracy_against_baseline(accuracy_metric: str, model: str) -> None:
    """Verify DUT accuracy metrics against baseline.

    Args:
        accuracy_metric (str): 'test_cases' or 'user_utterances'.
        model (str): Model used for DUT tests.
    """
    assert (
        results["accuracy"][accuracy_metric]
        >= BASELINE[model]["accuracy"][accuracy_metric]
    )


@pytest.mark.parametrize("f1_metric", {"macro", "micro", "weighted_average"})
def test_overall_f1_score_against_baseline(f1_metric: str, model: str) -> None:
    """Verify DUT overall F1 scores against baseline.

    Args:
        f1_metric (str): 'macro', 'micro' or 'weighted' f1-score
        model (str): Model used for DUT tests.
    """
    assert results["f1_score"][f1_metric] >= BASELINE[model]["f1_score"][f1_metric]


@pytest.mark.parametrize(
    "command",
    {
        "start_flow",
        "set_slot",
        "knowledge",
        "clarify",
        "chitchat",
        "cancel_flow",
        "human_handoff",
    },
)
def test_command_f1_score_against_baseline(command: str, model: str) -> None:
    """Verify f1-score of each command against baseline.

    Args:
        command (str): 'start_flow', 'set_slot', 'knowledge', 'clarify', 'chitchat', 'cancel_flow' or 'human_handoff'.
        model (str): Model used for DUT tests.
    """
    assert (
        results["command_metrics"][command]["f1_score"]
        >= BASELINE[model]["command_metrics"][command]["f1_score"]
    )
