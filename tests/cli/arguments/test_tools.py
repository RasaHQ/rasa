import argparse
from typing import List

import pytest

from rasa.cli.arguments.tools import MCP_TOOLS_DEFAULT_PORT


def test_default_tools_run_arguments(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests default settings for `rasa tools run` CLI command."""
    args = tools_parser.parse_args(["tools", "run"])

    assert args.project is None
    assert args.mode == "stdio"
    assert args.port == MCP_TOOLS_DEFAULT_PORT


def test_tools_run_with_mode_stdio(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --mode stdio`."""
    args = tools_parser.parse_args(["tools", "run", "--mode", "stdio"])

    assert args.mode == "stdio"


def test_tools_run_with_mode_http(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --mode http`."""
    args = tools_parser.parse_args(["tools", "run", "--mode", "http"])

    assert args.mode == "http"


def test_tools_run_with_invalid_mode(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests that invalid --mode value is rejected."""
    with pytest.raises(SystemExit):
        tools_parser.parse_args(["tools", "run", "--mode", "grpc"])


def test_tools_run_with_port(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --port 9999`."""
    args = tools_parser.parse_args(["tools", "run", "--port", "9999"])

    assert args.port == 9999


def test_tools_run_with_invalid_port(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests that non-integer --port value is rejected."""
    with pytest.raises(SystemExit):
        tools_parser.parse_args(["tools", "run", "--port", "not-a-number"])


def test_tools_run_with_project(
    tools_parser: argparse.ArgumentParser,
) -> None:
    """Tests `rasa tools run --project /some/path`."""
    args = tools_parser.parse_args(["tools", "run", "--project", "/some/path"])

    assert args.project == "/some/path"


@pytest.mark.parametrize(
    "input_args, expected_mode, expected_port, expected_project",
    [
        (
            [
                "tools",
                "run",
                "--mode",
                "http",
                "--port",
                "8080",
                "--project",
                "/my/bot",
            ],
            "http",
            8080,
            "/my/bot",
        ),
        (
            ["tools", "run", "--mode", "stdio", "--project", "."],
            "stdio",
            MCP_TOOLS_DEFAULT_PORT,
            ".",
        ),
        (
            ["tools", "run", "--port", "7777"],
            "stdio",
            7777,
            None,
        ),
    ],
)
def test_tools_run_combined_arguments(
    tools_parser: argparse.ArgumentParser,
    input_args: List[str],
    expected_mode: str,
    expected_port: int,
    expected_project: str,
) -> None:
    """Tests various combinations of `rasa tools run` arguments."""
    args = tools_parser.parse_args(input_args)

    assert args.mode == expected_mode
    assert args.port == expected_port
    assert args.project == expected_project
