import argparse
from typing import Callable
import pytest
from _pytest.pytester import RunResult

from rasa.__main__ import create_argument_parser


@pytest.fixture(autouse=True)
def parser() -> argparse.ArgumentParser:
    return create_argument_parser()


def test_data_convert_e2e_required_argument(parser: argparse.ArgumentParser):
    args = parser.parse_args(
        [
            "data",
            "convert",
            "e2e",
            "input_path.csv",
        ]
    )

    assert args.path == "input_path.csv"
    assert args.sheet_name is None


def test_data_convert_e2e_optional_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args(
        [
            "data",
            "convert",
            "e2e",
            "input_path.xls",
            "-o",
            "custom_output_dir",
            "--sheet-name",
            "Sheet1",
        ]
    )

    assert args.path == "input_path.xls"
    assert args.output == "custom_output_dir"
    assert args.sheet_name == "Sheet1"


def test_data_convert_e2e_no_argument(run: Callable[..., RunResult]):
    output = run("data", "convert", "e2e")
    error_line = (
        "rasa data convert e2e: error: the following arguments are required: path"
    )
    assert error_line in output.errlines


def test_data_convert_e2e_help(run: Callable[..., RunResult]):
    output = run("data", "convert", "e2e", "-h")

    help_text = """usage: rasa data convert e2e [-h] [-v] [-vv] [--quiet]
                             [--logging-config-file LOGGING_CONFIG_FILE]
                             [-o OUTPUT] [--sheet-name SHEET_NAME]
                             path"""

    lines = help_text.split("\n")
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help
