import argparse
from typing import Callable
from unittest.mock import patch

import pytest
from pytest import RunResult

from rasa.cli.llm_fine_tuning import (
    PARAMETERS_FILE,
    RESULT_SUMMARY_FILE,
    restricted_float,
    write_params,
    write_statistics,
    create_storage_context,
)
from rasa.llm_fine_tuning.storage import (
    StorageType,
    StorageContext,
    FileStorageStrategy,
)


def test_rasa_llm(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa llm [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE] {finetune}
    """
    lines = help_text.split("\n")

    output = run("llm", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help


def test_rasa_finetune_llm(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa llm finetune [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE] {prepare-data}
    """
    lines = help_text.split("\n")

    output = run("llm", "finetune", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help


def test_rasa_finetune_llm_prepare_data(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa llm finetune prepare-data [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE] [-o OUT]
    [--remote-storage REMOTE_STORAGE]
    [--num-rephrases {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
    25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49}]
    [--rephrase-config REPHRASE_CONFIG]
    [--train-frac TRAIN_FRAC]
    [--output-format [{alpaca,sharegpt,azure-gpt}]]
    [-m MODEL]
    [--endpoints ENDPOINTS]
    [path-to-e2e-test-cases]
    """
    lines = help_text.split("\n")

    output = run("llm", "finetune", "prepare-data", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help


@pytest.fixture
def args():
    mock_args = argparse.Namespace()
    mock_args.out = "output_test"
    mock_args.num_rephrases = 10
    mock_args.rephrase_config = "rephrasing_config.yaml"
    mock_args.train_frac = 0.8
    mock_args.output_format = "alpaca"
    mock_args.model = "dummy_model"
    mock_args.endpoints = "dummy_endpoints"
    mock_args.remote_storage = None
    mock_args.path_to_e2e_test_cases = "e2e_tests"
    return mock_args


def test_restricted_float():
    assert restricted_float(0.5) == 0.5
    assert restricted_float(1.0) == 1.0

    with pytest.raises(argparse.ArgumentTypeError):
        restricted_float(0.0)

    with pytest.raises(argparse.ArgumentTypeError):
        restricted_float(1.1)

    with pytest.raises(argparse.ArgumentTypeError):
        restricted_float("invalid")


@patch("rasa.shared.utils.yaml.write_yaml")
def test_write_params(mock_write_yaml, args):
    rephrase_config = {"some_key": "some_value"}

    write_params(args, rephrase_config, args.out)
    yaml_data = {
        "parameters": {
            "num_rephrases": args.num_rephrases,
            "rephrase_config": rephrase_config,
            "model": args.model,
            "endpoints": args.endpoints,
            "remote-storage": args.remote_storage,
            "train_frac": args.train_frac,
            "output_format": args.output_format,
            "out": args.out,
        }
    }
    mock_write_yaml.assert_called_once_with(yaml_data, f"{args.out}/{PARAMETERS_FILE}")


@patch("rasa.shared.utils.yaml.write_yaml")
def test_write_statistics(mock_write_yaml, args):
    statistics = {"stat1": 1, "stat2": 2}

    write_statistics(statistics, args.out)
    mock_write_yaml.assert_called_once_with(
        statistics, f"{args.out}/{RESULT_SUMMARY_FILE}"
    )


def test_create_storage_context():
    context = create_storage_context(StorageType.FILE, "output")

    assert isinstance(context, StorageContext) is True
    assert isinstance(context.strategy, FileStorageStrategy) is True
    assert context.strategy.output_dir == "output"
