import argparse
import platform
import sys
import os
import textwrap
from pathlib import Path
from typing import Any, Callable, List, Text, Dict
from unittest.mock import MagicMock, Mock, call, patch

import matplotlib.pyplot as plt
from _pytest.tmpdir import TempPathFactory
import pytest
from pytest import MonkeyPatch, RunResult
from structlog.testing import capture_logs

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
    _save_coverage_report,
    STATUS_PASSED,
    save_test_cases_to_yaml,
    _save_tested_commands_histogram,
)
from rasa.core.agent import Agent
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.e2e_test.assertions import AssertionFailure, FlowStartedAssertion
from rasa.e2e_test.constants import KEY_TEST_CASES
from rasa.e2e_test.e2e_test_case import Fixture, Metadata, TestCase, TestStep, TestSuite
from rasa.e2e_test.e2e_test_coverage_report import (
    FLOW_NAME_COL_NAME,
    NUM_STEPS_COL_NAME,
    MISSING_STEPS_COL_NAME,
    LINE_NUMBERS_COL_NAME,
    COVERAGE_COL_NAME,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.exceptions import RasaException
from rasa.shared.constants import DEFAULT_MODELS_PATH
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

    assert len(input_test_cases) == 9
    assert input_test_cases[0].file == str(e2e_input_folder / "e2e_one_test.yml")
    assert input_test_cases[1].file == str(
        e2e_input_folder / "e2e_one_test_with_fixtures.yml"
    )
    assert input_test_cases[2].file == str(
        e2e_input_folder / "e2e_test_case_with_slot_was_set.yml"
    )
    assert input_test_cases[3].file == str(e2e_input_folder / "e2e_test_cases.yml")
    assert input_test_cases[4].file == str(e2e_input_folder / "e2e_test_cases.yml")
    assert input_test_cases[5].file == str(
        e2e_input_folder / "e2e_test_cases_with_fixtures.yml"
    )
    assert input_test_cases[6].file == str(
        e2e_input_folder / "e2e_test_cases_with_fixtures.yml"
    )
    assert input_test_cases[7].file == str(
        e2e_input_folder / "e2e_test_cases_with_metadata.yml"
    )
    assert input_test_cases[8].file == str(
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


def test_validate_model_path_with_none(tmp_path: Path) -> None:
    parameter = "model"
    default = tmp_path / DEFAULT_MODELS_PATH
    with capture_logs() as logs:
        assert validate_model_path(None, parameter, default) == default

    log_msg = (
        f"Parameter '{parameter}' is not set. "
        f"Using default location '{default}' instead."
    )
    assert log_msg in logs[0]["message"]


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
                help="Set the remote location where your Rasa model is stored, "
                "e.g. on AWS.",
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


def test_save_coverage_report(tmp_path: Path):
    df_mock = MagicMock()
    df_mock.to_string.return_value = "Coverage Report"
    df_mock.to_csv.return_value = None
    df_mock.columns = [
        FLOW_NAME_COL_NAME,
        NUM_STEPS_COL_NAME,
        MISSING_STEPS_COL_NAME,
        LINE_NUMBERS_COL_NAME,
        COVERAGE_COL_NAME,
    ]
    df_mock.__getitem__.side_effect = lambda key: {
        FLOW_NAME_COL_NAME: ["flow1", "flow2", "Total"],
        NUM_STEPS_COL_NAME: [2, 3, 5],
        MISSING_STEPS_COL_NAME: [1, 2, 3],
        LINE_NUMBERS_COL_NAME: ["[[1-2], [3-4]]", "[[5-6], [7-8]]", ""],
        COVERAGE_COL_NAME: ["50.00%", "33.33%", "40.00%"],
    }[key]

    # Check the report is saved correctly
    output_filename = tmp_path / f"coverage_report_for_{STATUS_PASSED}_tests.csv"

    _save_coverage_report(df_mock, STATUS_PASSED, str(tmp_path))

    df_mock.to_csv.assert_called_with(str(output_filename), index=False)

    # Check the content of the DataFrame
    assert df_mock[FLOW_NAME_COL_NAME] == ["flow1", "flow2", "Total"]
    assert df_mock[NUM_STEPS_COL_NAME] == [2, 3, 5]
    assert df_mock[MISSING_STEPS_COL_NAME] == [1, 2, 3]
    assert df_mock[LINE_NUMBERS_COL_NAME] == [
        "[[1-2], [3-4]]",
        "[[5-6], [7-8]]",
        "",
    ]
    assert df_mock[COVERAGE_COL_NAME] == ["50.00%", "33.33%", "40.00%"]


def test_save_test_cases_to_yaml(tmp_path: Path):
    test_case = TestCase(
        "test_case",
        fixture_names=["fixture"],
        metadata_name="metadata",
        steps=[
            TestStep.from_dict({"user": "I need to check my balance!"}),
            TestStep.from_dict({"bot": "You have $40 in your account."}),
        ],
    )
    test_results = [TestResult(test_case, pass_status=True, difference=[])]
    test_suite = TestSuite(
        test_cases=[test_case],
        fixtures=[Fixture("fixture", {"key": "value"})],
        metadata=[Metadata("metadata", {"key": "value"})],
        stub_custom_actions={},
    )

    with capture_logs() as logs:
        save_test_cases_to_yaml(test_results, str(tmp_path), STATUS_PASSED, test_suite)
        output_file_path = str(tmp_path / f"{STATUS_PASSED}.yml")
        assert len(logs) == 2
        assert logs[0]["log_level"] == "info"
        assert logs[0]["message"] == (
            f"E2e tests with 'passed' status are written to file: '{output_file_path}'."
        )
        assert logs[1]["log_level"] == "info"
        assert logs[1]["message"] == (
            f"You can use the file: '{output_file_path}' in case you want to create "
            "training data for fine-tuning an LLM via 'rasa llm finetune prepare-data'."
        )

    actual_test_suite = read_test_cases(str(tmp_path / "passed.yml"))

    assert actual_test_suite.fixtures == test_suite.fixtures
    assert actual_test_suite.metadata == test_suite.metadata
    assert actual_test_suite.test_cases[0].name == test_suite.test_cases[0].name
    assert (
        actual_test_suite.test_cases[0].steps[0].actor
        == test_suite.test_cases[0].steps[0].actor
    )
    assert (
        actual_test_suite.test_cases[0].steps[0].text
        == test_suite.test_cases[0].steps[0].text
    )
    assert (
        actual_test_suite.test_cases[0].steps[1].actor
        == test_suite.test_cases[0].steps[1].actor
    )
    assert (
        actual_test_suite.test_cases[0].steps[1].text
        == test_suite.test_cases[0].steps[1].text
    )


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

    @patch("rasa.cli.e2e_test.plt.savefig")
    @patch("rasa.cli.e2e_test.structlogger.info")
    @patch(
        "rasa.cli.e2e_test.os.path.join", side_effect=lambda *args: os.path.join(*args)
    )
    def test_save_tested_commands_histogram(
        mock_join: MagicMock,
        mock_info: MagicMock,
        mock_savefig: MagicMock,
        tmpdir: TempPathFactory,
    ) -> None:
        # Define test data
        count_dict: Dict[str, int] = {"command1": 10, "command2": 5, "command3": 15}
        test_status: str = "passing"
        output_dir: str = str(tmpdir)
        output_filename: str = f"commands_histogram_for_{test_status}_tests.png"

        # Call the function
        _save_tested_commands_histogram(count_dict, test_status, output_dir)

        # Check that savefig was called with the correct path
        expected_output_file_path: str = os.path.join(output_dir, output_filename)
        mock_savefig.assert_called_once_with(expected_output_file_path)
        plt.close()  # Close the plot to clean up the state for other tests

        # Check that structlogger.info was called with the correct parameters
        mock_info.assert_called_once_with(
            "rasa.e2e_test._save_tested_commands_histogram",
            message=f"Commands histogram for {test_status} e2e tests "
            f"are written to '{expected_output_file_path}'.",
        )

        # Ensure that the file path was joined correctly
        mock_join.assert_called_once_with(output_dir, output_filename)

    @patch("rasa.cli.e2e_test.structlogger.info")
    @patch("rasa.cli.e2e_test.plt.savefig")
    def test_save_tested_commands_histogram_empty_dict(
        mock_savefig: MagicMock, mock_info: MagicMock
    ) -> None:
        # Define test data
        count_dict: Dict[str, int] = {}
        test_status: str = "failing"
        output_dir: str = "/some/fake/dir"

        # Call the function
        _save_tested_commands_histogram(count_dict, test_status, output_dir)

        # Check that savefig and info were never called because the dict is empty
        mock_savefig.assert_not_called()
        mock_info.assert_not_called()


def test_writing_test_suite():
    test_suite = read_test_cases(
        "data/end_to_end_testing_input_files/e2e_test_case_with_slot_was_set.yml"
    )
    test_suite_dict = test_suite.as_dict()

    test_case = TestCase.from_dict(test_suite_dict[KEY_TEST_CASES][0])

    assert test_case.steps == test_suite.test_cases[0].steps
