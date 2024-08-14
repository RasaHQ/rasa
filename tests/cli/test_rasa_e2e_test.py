import argparse
import logging
import platform
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, List, Text
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from pytest import LogCaptureFixture, MonkeyPatch, RunResult
from rasa.core.agent import Agent
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.e2e_test.assertions import AssertionFailure, FlowStartedAssertion
from rasa.exceptions import RasaException
from rasa.shared.constants import DEFAULT_MODELS_PATH
from rasa.shared.core.domain import Domain

from rasa.cli.e2e_test import (
    DEFAULT_E2E_INPUT_TESTS_PATH,
    DEFAULT_E2E_OUTPUT_TESTS_PATH,
    add_e2e_test_arguments,
    add_subparser,
    color_difference,
    execute_e2e_tests,
    is_test_case_file,
    print_failed_case,
    print_test_result,
    read_test_cases,
    split_into_passed_failed,
    transform_results_output_to_yaml,
    validate_model_path,
    validate_path_to_test_cases,
    write_test_results_to_file,
    extract_test_case_from_path,
    validate_test_case,
)
from rasa.e2e_test.e2e_test_case import Fixture, Metadata, TestCase, TestStep
from rasa.e2e_test.e2e_test_result import TestResult
from tests.e2e_test.test_e2e_test_runner import AsyncMock

SAVED_STDOUT = sys.stdout


def test_rasa_test_e2e_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa test e2e [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE] [--fail-fast]
                    [-o] [--remote-storage REMOTE_STORAGE] [-m MODEL]
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


def test_get_test_results_summary() -> None:
    results = [
        TestResult(TestCase("test_booking", []), True, []),
        TestResult(TestCase("test_mood_great", []), False, ["diff a"]),
        TestResult(TestCase("test_mood_sad", []), False, ["diff b"]),
    ]

    passed, failed = split_into_passed_failed(results)
    assert passed == results[:1]
    assert failed == results[1:]


def test_find_test_case_line_number(e2e_input_folder: Path) -> None:
    test_suite = read_test_cases(str(e2e_input_folder / "e2e_test_cases.yml"))
    input_test_cases = test_suite.test_cases

    assert len(input_test_cases) == 2
    assert input_test_cases[0].name == "test_booking"
    assert input_test_cases[0].line == 2

    assert input_test_cases[1].name == "test_mood_great"
    assert input_test_cases[1].line == 11


@pytest.mark.parametrize(
    "test_case_file,expected",
    [
        ("e2e_one_test.yml", True),
        ("e2e_test_cases.yml", True),
        ("e2e_test_cases.yml", True),
        ("e2e_empty.yml", True),
        ("not_a_test_file.yml", False),
        ("foo.bar", False),
    ],
)
def test_is_test_case_file(
    e2e_input_folder: Path, test_case_file: str, expected: bool
) -> None:
    assert is_test_case_file(str(e2e_input_folder / test_case_file)) == expected


def test_find_test_cases_handles_empty(e2e_input_folder: Path) -> None:
    test_suite = read_test_cases(str(e2e_input_folder / "e2e_empty.yml"))

    assert len(test_suite.test_cases) == 0


def test_find_test_sets_file(e2e_input_folder: Path) -> None:
    test_suite = read_test_cases(str(e2e_input_folder))
    input_test_cases = test_suite.test_cases

    assert len(input_test_cases) == 8
    assert input_test_cases[0].file == str(e2e_input_folder / "e2e_one_test.yml")
    assert input_test_cases[1].file == str(
        e2e_input_folder / "e2e_one_test_with_fixtures.yml"
    )
    assert input_test_cases[2].file == str(e2e_input_folder / "e2e_test_cases.yml")
    assert input_test_cases[3].file == str(e2e_input_folder / "e2e_test_cases.yml")
    assert input_test_cases[4].file == str(
        e2e_input_folder / "e2e_test_cases_with_fixtures.yml"
    )
    assert input_test_cases[5].file == str(
        e2e_input_folder / "e2e_test_cases_with_fixtures.yml"
    )
    assert input_test_cases[6].file == str(
        e2e_input_folder / "e2e_test_cases_with_metadata.yml"
    )
    assert input_test_cases[7].file == str(
        e2e_input_folder / "e2e_test_cases_with_metadata.yml"
    )


def test_color_difference_empty() -> None:
    assert list(color_difference([])) == []


def test_color_difference() -> None:
    assert list(color_difference(["+ Hello", "- World", "^  ++", "?   ~"])) == [
        "[green3]+ Hello[/green3]",
        "[red3]- World[/red3]",
        "[blue3]^  ++[/blue3]",
        "[grey37]?   ~[/grey37]",
    ]


@pytest.mark.parametrize(
    "input_tests_path, expected_results",
    [
        # Paths to files without global fixtures key
        ("data/end_to_end_testing_input_files/e2e_one_test.yml", []),
        ("data/end_to_end_testing_input_files/e2e_test_cases.yml", []),
        # Path to file with global fixtures key
        (
            "data/end_to_end_testing_input_files/e2e_test_cases_with_fixtures.yml",
            [
                Fixture(name="premium", slots_set={"membership_type": "premium"}),
                Fixture(name="standard", slots_set={"membership_type": "standard"}),
            ],
        ),
        # Path to directory with files with global fixtures key
        (
            "data/end_to_end_testing_input_files",
            [
                Fixture(name="premium", slots_set={"membership_type": "premium"}),
                Fixture(name="standard", slots_set={"membership_type": "standard"}),
            ],
        ),
    ],
)
def test_read_fixtures(input_tests_path: Text, expected_results: List[Fixture]) -> None:
    adjusted_path = Path(__file__).parent.parent.parent / input_tests_path
    test_suite = read_test_cases(str(adjusted_path))
    assert test_suite.fixtures == expected_results


@pytest.mark.parametrize(
    "input_tests_path, expected_results",
    [
        # Paths to files without global metadata key
        ("data/end_to_end_testing_input_files/e2e_one_test.yml", []),
        ("data/end_to_end_testing_input_files/e2e_test_cases.yml", []),
        # Path to file with global metadata key
        (
            "data/end_to_end_testing_input_files/e2e_test_cases_with_metadata.yml",
            [
                Metadata(
                    name="user_info",
                    metadata={"language": "English", "location": "Europe"},
                ),
                Metadata(name="device_info", metadata={"os": "linux"}),
            ],
        ),
        # Path to directory with files with global metadata key
        (
            "data/end_to_end_testing_input_files",
            [
                Metadata(
                    name="user_info",
                    metadata={"language": "English", "location": "Europe"},
                ),
                Metadata(name="device_info", metadata={"os": "linux"}),
            ],
        ),
    ],
)
def test_read_metadata(
    input_tests_path: Text, expected_results: List[Metadata]
) -> None:
    adjusted_path = Path(__file__).parent.parent.parent / input_tests_path
    test_suite = read_test_cases(str(adjusted_path))
    assert test_suite.metadata == expected_results


def test_transform_results_output_to_yaml() -> None:
    yaml_string = textwrap.dedent(
        """
    test_results:
    - name: happy path
      pass_status: true
      expected_steps:
       - user: "Hi!"
       - bot: "Hey! How are you?"
       - user: "I am feeling amazing."
       - bot: "Great, carry on!"

      difference: []
    - name: sad path with utter template (failing status)
      pass_status: false
      expected_steps:
       - user: "Hey"
       - utter: utter_greet
       - user: "I feel overwhelmed."
       - utter: utter_cheer_up
       - utter: utter_did_that_help

      difference:
       - '  user: Hey'
       - '  bot: Hey! How are you?'
       - '  user: I feel overwhelmed.'
       - '- bot: Great, carry on!'
       - '- * No Response *'
       - '+ bot: utter_cheer_up'
       - '+ bot: utter_did_that_help'"""
    )
    transformed_yaml_string = transform_results_output_to_yaml(yaml_string)

    expected_yaml_string = textwrap.dedent(
        """test_results:

- name: happy path
  pass_status: true
  expected_steps:
   - user: "Hi!"
   - bot: "Hey! How are you?"
   - user: "I am feeling amazing."
   - bot: "Great, carry on!"
  difference: []

- name: sad path with utter template (failing status)
  pass_status: false
  expected_steps:
   - user: "Hey"
   - utter: utter_greet
   - user: "I feel overwhelmed."
   - utter: utter_cheer_up
   - utter: utter_did_that_help
  difference:
   - '  user: Hey'
   - '  bot: Hey! How are you?'
   - '  user: I feel overwhelmed.'
   - '- bot: Great, carry on!'
   - '- * No Response *'
   - '+ bot: utter_cheer_up'
   - '+ bot: utter_did_that_help'"""
    )

    assert transformed_yaml_string == expected_yaml_string


def test_validate_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.tar.gz"
    model_path.touch()
    assert validate_model_path(str(model_path), "model", DEFAULT_MODELS_PATH) == str(
        model_path
    )


def test_validate_model_path_path_not_exists(tmp_path: Path) -> None:
    model_path = tmp_path / "model.tar.gz"
    default = tmp_path / DEFAULT_MODELS_PATH
    match_msg = (
        f"The provided model path '{model_path!s}' could not be found. "
        f"Using default location '{default!s}' instead."
    )
    if platform.system() == "Windows":
        # Windows uses backslashes in paths
        match_msg = match_msg.replace("\\", "\\\\")

    with pytest.warns(UserWarning, match=match_msg):
        assert validate_model_path(str(model_path), "model", default) == default


def test_validate_model_path_with_none(
    caplog: LogCaptureFixture, tmp_path: Path
) -> None:
    parameter = "model"
    default = tmp_path / DEFAULT_MODELS_PATH
    with caplog.at_level(logging.INFO):
        assert validate_model_path(None, parameter, default) == default

    log_msg = (
        f"Parameter '{parameter}' is not set. "
        f"Using default location '{default}' instead."
    )
    assert log_msg in caplog.text


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
    )

    captured = capsys.readouterr()

    assert (
        f"Overall results have been saved at path: {cli_args.e2e_results}."
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
    )
    captured = capsys.readouterr()

    assert (
        f"Overall results have been saved at path: {cli_args.e2e_results}."
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
                help="Set the remote location where your Rasa model is stored, e.g. on AWS.",  # noqa: E501
            ),
        ],
    )


@pytest.mark.parametrize(
    "results, output_file",
    [
        (
            [
                TestResult(
                    test_case=TestCase(
                        name="some test case", steps=[TestStep(actor="some actor")]
                    ),
                    pass_status=True,
                    difference=["something", "different"],
                ),
            ],
            "some/file/path/e2e_one_test.yml",
        ),
    ],
)
@patch("rasa.shared.utils.cli.print_info")
@patch("rasa.utils.io.write_yaml")
@patch("rasa.cli.e2e_test.Path")
def test_write_test_results_to_file(
    mock_path: MagicMock,
    mock_write_yaml: MagicMock,
    mock_print_info: MagicMock,
    results: List[TestResult],
    output_file: Text,
) -> None:
    path_instance = mock_path.return_value
    path_instance.touch = MagicMock()
    write_test_results_to_file(results, output_file)

    mock_path.assert_called_with(output_file)

    expected_result = {
        "test_results": [test_result.as_dict() for test_result in results]
    }
    mock_write_yaml.assert_called_with(
        expected_result, target=output_file, transform=transform_results_output_to_yaml
    )
    mock_print_info.assert_called_with(
        f"Overall results have been saved at path: {output_file}."
    )


def test_execute_e2e_tests_with_agent_not_ready(
    tmp_path: Path,
    caplog: LogCaptureFixture,
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

    logging_msg = (
        "Agent needs to be prepared before usage. "
        "Please check that the agent was able to "
        "load the trained model."
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            execute_e2e_tests(cli_args)

        assert logging_msg in [record.message for record in caplog.records]


def test_validate_path_to_test_cases(tmp_path: Path) -> None:
    """Test that a path to test cases which doesn't exist is validated correctly.

    The tested function should raise a UserWarning and exit the program.
    """
    path_to_test_cases = tmp_path / "test_cases.yml"

    match_msg = f"Path to test cases does not exist: {path_to_test_cases!s}."

    if platform.system() == "Windows":
        # Windows uses backslashes in paths
        match_msg = match_msg.replace("\\", "\\\\")

    with pytest.warns(UserWarning, match=match_msg):
        with pytest.raises(SystemExit):
            validate_path_to_test_cases(str(path_to_test_cases))


@pytest.mark.parametrize(
    "full_path_to_test_case, expected_path_to_test_cases, expected_test_case",
    [
        ("some/file/path/e2e_one_test.yml", "some/file/path/e2e_one_test.yml", ""),
        (
            "some/file/path/e2e_one_test.yml::test_case1",
            "some/file/path/e2e_one_test.yml",
            "test_case1",
        ),
    ],
)
def test_extract_test_case_from_path(
    full_path_to_test_case: str,
    expected_path_to_test_cases: Path,
    expected_test_case: str,
) -> None:
    """Test that test case are correctly extracted from the path to test cases."""
    path_to_test_cases, test_case = extract_test_case_from_path(
        str(full_path_to_test_case)
    )
    assert path_to_test_cases == expected_path_to_test_cases
    assert test_case == expected_test_case


def test_validate_test_case() -> None:
    """Test that a path to a test case which doesn't exist is validated correctly.

    The tested function should raise a UserWarning and exit the program.
    """
    test_case = "test_case1"
    match_msg = f"Test case does not exist: {test_case!s}."

    with pytest.warns(UserWarning, match=match_msg):
        with pytest.raises(SystemExit):
            validate_test_case(test_case, [])


@pytest.mark.parametrize(
    "test_cases_file, test_case, number_of_steps",
    [
        ("e2e_test_cases.yml", "test_booking", 6),
        ("e2e_test_cases.yml", "test_mood_great", 4),
        ("", "test_booking", 6),
        ("", "test_standard_booking", 8),
    ],
)
def test_read_single_test_case(
    e2e_input_folder: Path, test_cases_file: str, test_case: str, number_of_steps: int
) -> None:
    path_to_test_case = str(e2e_input_folder / f"{test_cases_file}::{test_case}")
    test_suite = read_test_cases(path_to_test_case)
    input_test_cases = test_suite.test_cases

    assert len(input_test_cases) == 1
    assert input_test_cases[0].name == test_case
    assert len(input_test_cases[0].steps) == number_of_steps


@pytest.mark.parametrize(
    "test_result",
    [
        TestResult(
            test_case=TestCase(
                name="some test case",
                steps=[TestStep(actor="some actor")],
                file="test.yaml",
            ),
            pass_status=False,
            difference=[],
            assertion_failure=AssertionFailure(
                assertion=FlowStartedAssertion(flow_id="test_flow_id"),
                error_message="Test error message",
                actual_events_transcript=["test_event"],
            ),
            error_line=1,
        ),
    ],
)
@patch("rasa.shared.utils.cli.print_error")
def test_print_failed_case(
    mock_print_error: MagicMock,
    test_result: TestResult,
) -> None:
    print_failed_case(test_result)

    mock_print_error.assert_any_call("Mismatch starting at test.yaml:1: \n")
    mock_print_error.assert_any_call(
        "Assertion type 'flow_started' failed with this error message: "
        "Test error message\n"
    )
    mock_print_error.assert_any_call("Actual events transcript:\n")
    mock_print_error.assert_called_with("test_event")


@patch("rasa.cli.e2e_test.print_aggregate_stats")
def test_print_test_result_with_aggregate_stats(
    mock_print_aggregate_stats: MagicMock,
) -> None:
    accuracy_calculations = [Mock()]

    with pytest.raises(SystemExit):
        print_test_result(
            [
                TestResult(
                    test_case=TestCase(
                        name="some test case",
                        steps=[TestStep(actor="some actor")],
                        file="test.yaml",
                    ),
                    pass_status=True,
                    difference=[],
                )
            ],
            [],
            fail_fast=False,
            accuracy_calculations=accuracy_calculations,
        )

    mock_print_aggregate_stats.assert_called_with(accuracy_calculations)


@patch("rasa.cli.e2e_test.print_aggregate_stats")
def test_print_test_result_without_aggregate_stats(
    mock_print_aggregate_stats: MagicMock,
) -> None:
    accuracy_calculations = []

    with pytest.raises(SystemExit):
        print_test_result(
            [
                TestResult(
                    test_case=TestCase(
                        name="some test case",
                        steps=[TestStep(actor="some actor")],
                        file="test.yaml",
                    ),
                    pass_status=True,
                    difference=[],
                )
            ],
            [],
            fail_fast=False,
            accuracy_calculations=accuracy_calculations,
        )

    mock_print_aggregate_stats.assert_not_called()
