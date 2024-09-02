import argparse
import sys
from pathlib import Path
from typing import Any, Callable
from unittest.mock import MagicMock, call

import pytest
from pytest import MonkeyPatch, RunResult
from structlog.testing import capture_logs

import rasa.cli.utils
from rasa.cli.e2e_test import (
    add_e2e_test_arguments,
    add_subparser,
    execute_e2e_tests,
)
from rasa.core.agent import Agent
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.e2e_test.constants import (
    DEFAULT_E2E_INPUT_TESTS_PATH,
    DEFAULT_E2E_OUTPUT_TESTS_PATH,
    STATUS_FAILED,
    STATUS_PASSED,
)
from rasa.e2e_test.e2e_test_case import TestCase
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.e2e_test.utils.io import read_test_cases
from rasa.exceptions import RasaException
from rasa.shared.core.domain import Domain
from tests.e2e_test.test_e2e_test_runner import AsyncMock

SAVED_STDOUT = sys.stdout


def test_rasa_test_e2e_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa test e2e [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE] [--fail-fast]
                    [-o] [--remote-storage REMOTE_STORAGE]
                    [--coverage-report]
                    [--coverage-output-path COVERAGE_OUTPUT_PATH] [-m MODEL]
                    [--endpoints ENDPOINTS]
                    [path-to-test-cases]

                   Runs end-to-end testing."""
    lines = help_text.split("\n")

    output = run("test", "e2e", "--help")

    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_add_subparser_fails_if_not_found() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    with pytest.raises(RasaException):
        add_subparser(subparsers, [])


def test_execute_e2e_tests_fail_fast_true(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: Any,
    e2e_input_folder,
) -> None:
    cli_args = argparse.Namespace()
    cli_args.endpoints = str(tmp_path / "endpoints.yml")
    cli_args.model = str(tmp_path / "model.tar.gz")
    setattr(
        cli_args,
        "path-to-test-cases",
        e2e_input_folder,
    )
    cli_args.fail_fast = True
    cli_args.e2e_results = str(tmp_path / "e2e_results.yml")
    cli_args.remote_storage = None
    cli_args.coverage_report = False

    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        domain = Domain.empty()
        self.agent = Agent(
            domain=domain, tracker_store=InMemoryTrackerStore(domain=domain)
        )

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    run_tests_mock = AsyncMock()
    path_to_test_cases = getattr(cli_args, "path-to-test-cases")

    run_tests_mock.return_value = [
        TestResult(TestCase("test_failure", [], path_to_test_cases, 1), False, []),
        TestResult(TestCase("test_success", [], path_to_test_cases, 10), True, []),
    ]
    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.run_tests", run_tests_mock
    )

    with pytest.raises(SystemExit):
        execute_e2e_tests(cli_args)

    test_suite = read_test_cases(path_to_test_cases)

    run_tests_mock.assert_called_once_with(
        test_suite.test_cases,
        test_suite.fixtures,
        cli_args.fail_fast,
        input_metadata=test_suite.metadata,
        coverage=cli_args.coverage_report,
    )

    captured = capsys.readouterr()

    passed_results_path = rasa.cli.utils.get_e2e_results_file_name(
        Path(cli_args.e2e_results), STATUS_PASSED
    )
    failed_results_path = rasa.cli.utils.get_e2e_results_file_name(
        Path(cli_args.e2e_results), STATUS_FAILED
    )

    assert (
        f"Passing test results have been saved at path: {passed_results_path}."
        in captured.out
    )
    assert (
        f"Failing test results have been saved at path: {failed_results_path}."
        in captured.out
    )
    assert f"'test_failure' in {path_to_test_cases}:1 failed" in captured.out
    assert f"FAILED {path_to_test_cases}::test_failure" in captured.out
    assert "stopping after 1 failure" in captured.out


def test_execute_e2e_tests_fail_fast_false(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: Any,
    e2e_input_folder: Path,
) -> None:
    cli_args = argparse.Namespace()
    cli_args.endpoints = str(tmp_path / "endpoints.yml")
    cli_args.model = str(tmp_path / "model.tar.gz")
    setattr(
        cli_args,
        "path-to-test-cases",
        e2e_input_folder,
    )
    cli_args.fail_fast = False
    cli_args.e2e_results = str(tmp_path / "e2e_results.yml")
    cli_args.remote_storage = None
    cli_args.coverage_report = False

    def mock_init(self: Any, *args: Any, **kwargs: Any) -> None:
        domain = Domain.empty()
        self.agent = Agent(
            domain=domain, tracker_store=InMemoryTrackerStore(domain=domain)
        )

    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.__init__", mock_init
    )

    run_tests_mock = AsyncMock()
    path_to_test_cases = getattr(cli_args, "path-to-test-cases")
    run_tests_mock.return_value = [
        TestResult(TestCase("test_failure", [], path_to_test_cases, 1), False, []),
        TestResult(TestCase("test_success", [], path_to_test_cases, 10), True, []),
    ]
    monkeypatch.setattr(
        "rasa.e2e_test.e2e_test_runner.E2ETestRunner.run_tests", run_tests_mock
    )

    with pytest.raises(SystemExit):
        execute_e2e_tests(cli_args)

    test_suite = read_test_cases(path_to_test_cases)

    run_tests_mock.assert_called_once_with(
        test_suite.test_cases,
        test_suite.fixtures,
        cli_args.fail_fast,
        input_metadata=test_suite.metadata,
        coverage=cli_args.coverage_report,
    )
    captured = capsys.readouterr()

    passed_results_path = rasa.cli.utils.get_e2e_results_file_name(
        Path(cli_args.e2e_results), STATUS_PASSED
    )
    failed_results_path = rasa.cli.utils.get_e2e_results_file_name(
        Path(cli_args.e2e_results), STATUS_FAILED
    )

    assert (
        f"Passing test results have been saved at path: {passed_results_path}."
        in captured.out
    )
    assert (
        f"Failing test results have been saved at path: {failed_results_path}."
        in captured.out
    )
    assert f"'test_failure' in {path_to_test_cases}:1 failed" in captured.out
    assert f"FAILED {path_to_test_cases}::test_failure" in captured.out
    assert "stopping after 1 failure" not in captured.out
    assert "1 failed, 1 passed" in captured.out


def test_e2e_cli_add_e2e_test_arguments(monkeypatch: MonkeyPatch) -> None:
    mock_e2e_arguments = MagicMock()
    mock_e2e_arguments.add_argument = MagicMock()

    mock_add_argument_group = MagicMock()
    mock_add_argument_group.return_value = mock_e2e_arguments

    mock_add_argument = MagicMock()

    parser = argparse.ArgumentParser()
    monkeypatch.setattr(parser, "add_argument_group", mock_add_argument_group)
    monkeypatch.setattr(parser, "add_argument", mock_add_argument)

    add_e2e_test_arguments(parser)

    mock_e2e_arguments.add_argument.assert_has_calls(
        [
            call(
                "path-to-test-cases",
                nargs="?",
                type=str,
                help="Input file or folder containing end-to-end test cases.",
                default=DEFAULT_E2E_INPUT_TESTS_PATH,
            ),
            call(
                "--fail-fast",
                action="store_true",
                help="Fail the test suite as soon as a unit test fails.",
            ),
        ]
    )

    mock_add_argument.assert_has_calls(
        [
            call(
                "-o",
                "--e2e-results",
                action="store_const",
                const=DEFAULT_E2E_OUTPUT_TESTS_PATH,
                help="Results file containing end-to-end testing summary.",
            ),
            call(
                "--remote-storage",
                help="Set the remote location where your Rasa model is stored, "
                "e.g. on AWS.",
            ),
        ],
    )


def test_execute_e2e_tests_with_agent_not_ready(
    tmp_path: Path,
    e2e_input_folder: Path,
) -> None:
    cli_args = argparse.Namespace()
    cli_args.endpoints = str(tmp_path / "endpoints.yml")
    cli_args.model = str(tmp_path / "model.tar.gz")
    setattr(
        cli_args,
        "path-to-test-cases",
        e2e_input_folder,
    )
    cli_args.fail_fast = False
    cli_args.e2e_results = str(tmp_path / "e2e_results.yml")
    cli_args.remote_storage = None
    cli_args.coverage_report = False

    logging_msg = (
        "Agent needs to be prepared before usage. "
        "Please check that the agent was able to "
        "load the trained model."
    )

    with capture_logs() as logs:
        with pytest.raises(SystemExit):
            execute_e2e_tests(cli_args)

        assert logging_msg in [
            record["message"] for record in logs if "message" in record
        ]
