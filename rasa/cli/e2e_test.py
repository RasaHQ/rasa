import argparse
import asyncio
import math
import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional, Text, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
import rich
import structlog
from rich.table import Table

import rasa.cli.arguments.run
import rasa.cli.utils
import rasa.shared.data
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.utils.io
from rasa.cli import SubParsersAction
from rasa.cli.arguments.default_arguments import add_endpoint_param, add_model_param
from rasa.core.exceptions import AgentNotReady
from rasa.core.utils import AvailableEndpoints
from rasa.e2e_test.aggregate_test_stats_calculator import (
    AccuracyCalculation,
    AggregateTestStatsCalculator,
)
from rasa.e2e_test.constants import SCHEMA_FILE_PATH, KEY_TEST_CASE, KEY_TEST_CASES
from rasa.e2e_test.e2e_test_case import (
    KEY_FIXTURES,
    KEY_METADATA,
    KEY_STUB_CUSTOM_ACTIONS,
    Fixture,
    Metadata,
    StubCustomAction,
    TestCase,
    TestSuite,
)
from rasa.e2e_test.e2e_test_coverage_report import (
    create_coverage_report,
    extract_tested_commands,
)
from rasa.e2e_test.e2e_test_result import TestResult
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.exceptions import RasaException
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH, DEFAULT_MODELS_PATH
from rasa.shared.utils.yaml import (
    parse_raw_yaml,
    read_schema_file,
    validate_yaml_data_using_schema_with_assertions,
    is_key_in_yaml,
)
from rasa.utils.beta import BetaNotEnabledException, ensure_beta_feature_is_enabled

DEFAULT_E2E_INPUT_TESTS_PATH = "tests/e2e_test_cases.yml"
DEFAULT_E2E_OUTPUT_TESTS_PATH = "tests/e2e_results.yml"
DEFAULT_COVERAGE_OUTPUT_PATH = "e2e_coverage_results"

# Test status
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"

RASA_PRO_BETA_E2E_ASSERTIONS_ENV_VAR_NAME = "RASA_PRO_BETA_E2E_ASSERTIONS"
RASA_PRO_BETA_FINE_TUNING_RECIPE_ENV_VAR_NAME = "RASA_PRO_BETA_FINE_TUNING_RECIPE"

structlogger = structlog.get_logger()


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add the e2e subparser to `rasa test`.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    for subparser in subparsers.choices.values():
        if subparser.prog == "rasa test":
            e2e_test_subparser = create_e2e_test_subparser(parents)

            for action in subparser._subparsers._actions:
                if action.choices is not None:
                    action.choices["e2e"] = e2e_test_subparser
                    return

    # If we get here, we couldn't hook the subparser to `rasa test`
    raise RasaException(
        "Hooking the e2e subparser to `rasa test` command "
        "could not be completed. Cannot run end-to-end testing."
    )


def create_e2e_test_subparser(
    parents: List[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Create e2e test subparser."""
    e2e_test_subparser = argparse.ArgumentParser(
        prog="rasa test e2e",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Runs end-to-end testing.",
    )

    e2e_test_subparser.set_defaults(func=execute_e2e_tests)

    add_e2e_test_arguments(e2e_test_subparser)
    add_model_param(e2e_test_subparser, add_positional_arg=False)
    add_endpoint_param(
        e2e_test_subparser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )

    return e2e_test_subparser


def add_e2e_test_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running E2E tests directly using `rasa e2e`."""
    e2e_arguments = parser.add_argument_group("Testing Settings")
    e2e_arguments.add_argument(
        "path-to-test-cases",
        nargs="?",
        type=str,
        help="Input file or folder containing end-to-end test cases.",
        default=DEFAULT_E2E_INPUT_TESTS_PATH,
    )
    e2e_arguments.add_argument(
        "--fail-fast",
        action="store_true",
        help="Fail the test suite as soon as a unit test fails.",
    )

    parser.add_argument(
        "-o",
        "--e2e-results",
        action="store_const",
        const=DEFAULT_E2E_OUTPUT_TESTS_PATH,
        help="Results file containing end-to-end testing summary.",
    )

    parser.add_argument(
        "--remote-storage",
        help="Set the remote location where your Rasa model is stored, e.g. on AWS.",
    )

    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate a coverage report on flow paths and commands covered in e2e "
        "tests.",
    )

    parser.add_argument(
        "--coverage-output-path",
        default=DEFAULT_COVERAGE_OUTPUT_PATH,
        help="Directory where to save coverage report to.",
    )


def split_into_passed_failed(
    results: List[TestResult],
) -> Tuple[List[TestResult], List[TestResult]]:
    """Get the summary of the test results.

    Args:
        results: List of test results.

    Returns:
        Tuple consisting of passed count, failed count and failed test cases.
    """
    passed_cases = [r for r in results if r.pass_status]
    failed_cases = [r for r in results if not r.pass_status]

    return passed_cases, failed_cases


def is_test_case_file(file_path: Union[Text, Path]) -> bool:
    """Check if file contains test cases.

    Args:
        file_path: Path of the file to check.

    Returns:
        `True` if the file contains test cases, `False` otherwise.
    """
    return rasa.shared.data.is_likely_yaml_file(file_path) and is_key_in_yaml(
        file_path, KEY_TEST_CASES
    )


def validate_path_to_test_cases(path: Text) -> None:
    """Validate that path to test cases exists."""
    if not Path(path).exists():
        rasa.shared.utils.io.raise_warning(
            f"Path to test cases does not exist: {path}. "
            f"Please provide a valid path to test cases. "
            f"Exiting...",
            UserWarning,
        )
        sys.exit(1)


@lru_cache(maxsize=1)
def extract_test_case_from_path(path: Text) -> Tuple[Text, Text]:
    """Extract test case from path if specified.

    Args:
        path: Path to the file or folder containing test cases.

    Returns:
        Tuple consisting of the path to test cases and the extracted test case name.
    """
    test_case_name = ""

    if "::" in str(path):
        splitted_path = path.split("::")
        test_case_name = splitted_path[-1]
        path = splitted_path[0]

    return path, test_case_name


def validate_test_case(test_case_name: Text, input_test_cases: List[TestCase]) -> None:
    """Validate that test case exists."""
    if test_case_name and not input_test_cases:
        rasa.shared.utils.io.raise_warning(
            f"Test case does not exist: {test_case_name}. "
            f"Please check for typos and provide a valid test case name. "
            f"Exiting...",
            UserWarning,
        )
        sys.exit(1)


def read_test_cases(path: Text) -> TestSuite:
    """Read test cases from the given path.

    Args:
        path: Path to the file or folder containing test cases.

    Returns:
        TestSuite.
    """
    path, test_case_name = extract_test_case_from_path(path)
    validate_path_to_test_cases(path)

    test_files = rasa.shared.data.get_data_files([path], is_test_case_file)
    e2e_test_schema = read_e2e_test_schema()

    input_test_cases = []
    fixtures: Dict[Text, Fixture] = {}
    metadata: Dict[Text, Metadata] = {}
    stub_custom_actions: Dict[Text, StubCustomAction] = {}

    beta_flag_verified = False

    for test_file in test_files:
        test_file_content = parse_raw_yaml(Path(test_file).read_text())

        validate_yaml_data_using_schema_with_assertions(
            yaml_data=test_file_content, schema_content=e2e_test_schema
        )

        test_cases_content = test_file_content.get(KEY_TEST_CASES) or []

        if test_case_name:
            test_cases = [
                TestCase.from_dict(test_case_dict, file=test_file)
                for test_case_dict in test_cases_content
                if test_case_name == test_case_dict.get(KEY_TEST_CASE)
            ]
        else:
            test_cases = [
                TestCase.from_dict(test_case_dict, file=test_file)
                for test_case_dict in test_cases_content
            ]

        beta_flag_verified = verify_beta_feature_flag_for_assertions(
            test_cases, beta_flag_verified
        )

        input_test_cases.extend(test_cases)
        fixtures_content = test_file_content.get(KEY_FIXTURES) or []
        for fixture in fixtures_content:
            fixture_obj = Fixture.from_dict(fixture_dict=fixture)

            # avoid adding duplicates from across multiple files
            if fixtures.get(fixture_obj.name) is None:
                fixtures[fixture_obj.name] = fixture_obj

        metadata_contents = test_file_content.get(KEY_METADATA) or []
        for metadata_content in metadata_contents:
            metadata_obj = Metadata.from_dict(metadata_dict=metadata_content)

            # avoid adding duplicates from across multiple files
            if metadata.get(metadata_obj.name) is None:
                metadata[metadata_obj.name] = metadata_obj

        stub_custom_actions_contents = (
            test_file_content.get(KEY_STUB_CUSTOM_ACTIONS) or {}
        )
        for action_name, mock_data in stub_custom_actions_contents.items():
            stub_custom_actions[action_name] = StubCustomAction.from_dict(
                action_name=action_name,
                mock_data=mock_data,
            )

    validate_test_case(test_case_name, input_test_cases)
    return TestSuite(
        input_test_cases,
        list(fixtures.values()),
        list(metadata.values()),
        stub_custom_actions,
    )


def execute_e2e_tests(args: argparse.Namespace) -> None:
    """Run the end-to-end tests.

    Args:
        args: Commandline arguments.
    """
    args.endpoints = rasa.cli.utils.get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

    # Ignore all endpoints apart from action server, model, nlu and nlg
    # to ensure InMemoryTrackerStore is being used instead of production
    # tracker store
    endpoints.tracker_store = None
    endpoints.lock_store = None
    endpoints.event_broker = None

    if endpoints.model is None:
        args.model = validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)

    path_to_test_cases = getattr(
        args, "path-to-test-cases", DEFAULT_E2E_INPUT_TESTS_PATH
    )

    test_suite = read_test_cases(path_to_test_cases)

    if endpoints.action and test_suite.stub_custom_actions:
        endpoints.action.kwargs[KEY_STUB_CUSTOM_ACTIONS] = (
            test_suite.stub_custom_actions
        )

    test_case_path, _ = extract_test_case_from_path(path_to_test_cases)

    try:
        test_runner = E2ETestRunner(
            remote_storage=args.remote_storage,
            model_path=args.model,
            model_server=endpoints.model,
            endpoints=endpoints,
            test_case_path=Path(test_case_path),
        )
    except AgentNotReady as error:
        structlogger.error(
            "rasa.e2e_test.execute_e2e_tests.agent_not_ready", message=error.message
        )
        sys.exit(1)

    results = asyncio.run(
        test_runner.run_tests(
            test_suite.test_cases,
            test_suite.fixtures,
            args.fail_fast,
            input_metadata=test_suite.metadata,
            coverage=args.coverage_report,
        )
    )

    if args.e2e_results is not None:
        write_test_results_to_file(results, args.e2e_results)

    passed, failed = split_into_passed_failed(results)
    aggregate_stats_calculator = AggregateTestStatsCalculator(
        passed_results=passed, failed_results=failed, test_cases=test_suite.test_cases
    )
    accuracy_calculations = aggregate_stats_calculator.calculate()

    if args.coverage_report and test_runner.agent.processor:
        ensure_beta_feature_is_enabled(
            "LLM fine-tuning recipe",
            env_flag=RASA_PRO_BETA_FINE_TUNING_RECIPE_ENV_VAR_NAME,
        )
        coverage_output_path = args.coverage_output_path
        rasa.shared.utils.io.create_directory(coverage_output_path)
        flows = asyncio.run(test_runner.agent.processor.get_flows())

        for results, status in [(passed, STATUS_PASSED), (failed, STATUS_FAILED)]:
            report = create_coverage_report(flows, results)
            _save_coverage_report(report, status, coverage_output_path)
            tested_commands = extract_tested_commands(results)
            _save_tested_commands_histogram(
                tested_commands, status, coverage_output_path
            )
            save_test_cases_to_yaml(results, coverage_output_path, status, test_suite)

        rasa.shared.utils.cli.print_info(
            f"Coverage data and result is written to {coverage_output_path}."
        )

    print_test_result(
        passed, failed, args.fail_fast, accuracy_calculations=accuracy_calculations
    )


def _save_coverage_report(
    report: Optional[pd.DataFrame], test_status: str, output_dir: str
) -> None:
    if report is None:
        return

    if report is not None and not report.empty:
        if test_status == STATUS_PASSED:
            rasa.shared.utils.cli.print_success(report.to_string(index=False))
        else:
            rasa.shared.utils.cli.print_error(report.to_string(index=False))

    output_filename = f"coverage_report_for_{test_status}_tests.csv"
    report.to_csv(
        os.path.join(output_dir, output_filename),
        index=False,
    )
    structlogger.info(
        "rasa.e2e_test.save_coverage_report",
        message=f"Coverage result for {test_status} e2e tests"
        f"is written to '{output_filename}'.",
    )


def write_test_results_to_file(results: List[TestResult], output_file: Text) -> None:
    """Write test results to a file.

    Args:
        results: List of test results.
        output_file: Path to the output file.
    """
    Path(output_file).touch()

    data = {"test_results": [test_result.as_dict() for test_result in results]}

    rasa.utils.io.write_yaml(
        data, target=output_file, transform=transform_results_output_to_yaml
    )

    rasa.shared.utils.cli.print_info(
        f"Overall results have been saved at path: {output_file}."
    )


def transform_results_output_to_yaml(yaml_string: Text) -> Text:
    """Transform the output of the YAML writer to make it more readable.

    Args:
        yaml_string: The YAML string to transform.

    Returns:
        The transformed YAML string.
    """
    result = []
    for s in yaml_string.splitlines(True):
        if s.startswith("- name"):
            result.append("\n")
            result.append(s)
        elif s.startswith("\n"):
            result.append(s.strip())
        else:
            result.append(s)
    return "".join(result)


def color_difference(diff: List[Text]) -> Generator[Text, None, None]:
    """Colorize the difference between two strings.

    Example:
        >>> color_difference(["+ Hello", "- World"])
        ["<ansigreen>+ Hello</ansigreen>", "<ansired>- World</ansired>"]

    Args:
        diff: List of lines of the diff.

    Returns:
    Generator of colored lines.
    """
    for line in diff:
        if line.startswith("+"):
            yield "[green3]" + line + "[/green3]"
        elif line.startswith("-"):
            yield "[red3]" + line + "[/red3]"
        elif line.startswith("^"):
            yield "[blue3]" + line + "[/blue3]"
        elif line.startswith("?"):
            yield "[grey37]" + line + "[/grey37]"
        else:
            yield line


def print_failed_case(fail: TestResult) -> None:
    """Print the details of a failed test case.

    Example:
        >>> print_failed_case(TestResult(TestCase("test", "test.md"), 1,
        ...                  ["- Hello", "+ World"]))
        ---------------------- test in test.md failed ----------------------
        Mismatch starting at test.md:1:
        <ansired>- Hello</ansired>
        <ansigreen>+ World</ansigreen>
    """
    fail_headline = (
        f"'{fail.test_case.name}' in {fail.test_case.file_with_line()} failed"
    )
    rasa.shared.utils.cli.print_error(
        f"{rasa.shared.utils.cli.pad(fail_headline, char='-')}\n"
    )
    rasa.shared.utils.cli.print_error(
        f"Mismatch starting at {fail.test_case.file}:{fail.error_line}: \n"
    )
    if fail.difference:
        rich.print(("\n".join(color_difference(fail.difference))))

    if fail.assertion_failure:
        rasa.shared.utils.cli.print_error(
            f"Assertion type '{fail.assertion_failure.assertion.type()}' failed "
            f"with this error message: {fail.assertion_failure.error_message}\n"
        )
        rasa.shared.utils.cli.print_error("Actual events transcript:\n")
        rasa.shared.utils.cli.print_error(
            "\n".join(fail.assertion_failure.actual_events_transcript)
        )


def print_test_summary(failed: List[TestResult]) -> None:
    """Print the summary of the test run.

    Example:
        >>> print_test_summary([TestResult(TestCase("test", "test.md"), 1,
        ...                  ["- Hello", "+ World"])])
        =================== short test summary info ===================
        FAILED test.md::test
    """
    rasa.shared.utils.cli.print_info(
        rasa.shared.utils.cli.pad("short test summary info")
    )

    for f in failed:
        rasa.shared.utils.cli.print_error(
            f"FAILED {f.test_case.file}::{f.test_case.name}"
        )


def print_final_line(
    passed: List[TestResult], failed: List[TestResult], has_failed: bool
) -> None:
    """Print the final line of the test output.

    Args:
        passed: List of passed test cases.
        failed: List of failed test cases.
        has_failed: Boolean, true if the test run has failed.
    """
    final_line_color = "green3" if not has_failed else "red3"

    width = shutil.get_terminal_size((80, 20)).columns

    # calculate the length of the text - this is a bit hacky but works
    text_lengt = (
        math.ceil(len(passed) / 10)  # length of the number of passed tests
        + math.ceil(len(failed) / 10)  # length of the number of failed tests
        + 18  # length of the text "  failed, passed  "
    )
    # we can't use the padding function here as the text contains html tags
    # which are not taken into account when calculating the length
    padding = max(6, width - text_lengt)
    pre_pad = "=" * max(3, padding // 2)
    post_pad = "=" * max(3, math.ceil(padding / 2))
    rich.print(
        f"[{final_line_color}]{pre_pad} "
        f"[bold red3]{len(failed)} failed[/bold red3]"
        f"[bright_white], [/bright_white]"
        f"[bold green3]{len(passed)} passed[/bold green3]"
        f" {post_pad}[/{final_line_color}]"
    )


def print_aggregate_stats(accuracy_calculations: List[AccuracyCalculation]) -> None:
    """Print the aggregate statistics of the test run."""
    rasa.shared.utils.cli.print_info(
        rasa.shared.utils.cli.pad("Accuracy By Assertion Type")
    )
    table = Table()
    table.add_column("Assertion Type", justify="center", style="cyan")
    table.add_column("Accuracy", justify="center", style="cyan")

    for accuracy_calculation in accuracy_calculations:
        table.add_row(
            accuracy_calculation.assertion_type, f"{accuracy_calculation.accuracy:.2%}"
        )

    rich.print(table)


def print_test_result(
    passed: List[TestResult],
    failed: List[TestResult],
    fail_fast: bool = False,
    **kwargs: Any,
) -> None:
    """Print the result of the test run.

    Args:
        passed: List of passed test cases.
        failed: List of failed test cases.
        fail_fast: If true, stop after the first failure.
        **kwargs: additional arguments
    """
    if failed:
        # print failure headline
        print("\n")
        rich.print(f"[bold]{rasa.shared.utils.cli.pad('FAILURES', char='=')}[/bold]")

    # print failed test_Case
    for fail in failed:
        print_failed_case(fail)

    accuracy_calculations = kwargs.get("accuracy_calculations", [])
    if accuracy_calculations:
        print_aggregate_stats(accuracy_calculations)

    print_test_summary(failed)

    if fail_fast:
        rasa.shared.utils.cli.print_error(
            rasa.shared.utils.cli.pad("stopping after 1 failure", char="!")
        )
        has_failed = True
    elif len(failed) + len(passed) == 0:
        # no tests were run, print error
        rasa.shared.utils.cli.print_error(
            rasa.shared.utils.cli.pad("no test cases found", char="!")
        )
        print_e2e_help()
        has_failed = True
    elif failed:
        has_failed = True
    else:
        has_failed = False

    print_final_line(passed, failed, has_failed=has_failed)
    sys.exit(1 if has_failed else 0)


def print_e2e_help() -> None:
    """Print help guiding users how to write e2e tests."""
    rasa.shared.utils.cli.print_info(
        dedent(
            """\
        To start using e2e tests create a yaml file in a test directory, e.g.
        'tests/test_cases.yml'. You can find example test cases in the starter
        pack at

            https://github.com/RasaHQ/starter-pack-intentless-policy#testing-the-policy

        Here is an example of a test case in yaml format:

            test_cases:
            - test_case: "test_greet"
              steps:
              - user: "hello there!"
              - bot: "Hey! How are you?"

        To run the e2e tests, execute:
            >>> rasa test e2e <path-to-test-cases>
    """
        )
    )


def validate_model_path(
    model_path: Optional[Text], parameter: Text, default: Text
) -> Text:
    """Validate the model path.

    Args:
        model_path: Path to the model.
        parameter: Name of the parameter.
        default: Default path to the model.

    Returns:
    Path to the model.
    """
    if model_path and Path(model_path).exists():
        return model_path

    if model_path and not Path(model_path).exists():
        rasa.shared.utils.io.raise_warning(
            f"The provided model path '{model_path}' could not be found. "
            f"Using default location '{default}' instead.",
            UserWarning,
        )

    elif model_path is None:
        structlogger.info(
            "rasa.e2e_test.validate_model_path",
            message=f"Parameter '{parameter}' is not set. "
            f"Using default location '{default}' instead.",
        )

    Path(default).mkdir(exist_ok=True)
    return default


def read_e2e_test_schema() -> Union[List[Any], Dict[Text, Any]]:
    """Read the schema for the e2e test files.

    Returns:
        The content of the schema.
    """
    return read_schema_file(SCHEMA_FILE_PATH)


def save_test_cases_to_yaml(
    test_results: List[TestResult],
    output_dir: str,
    status: str,
    test_suite: TestSuite,
) -> None:
    """Extracts TestCases from a list of TestResults and saves them to a YAML file."""
    if not test_results:
        return

    test_cases = [result.test_case for result in test_results]
    new_test_suite = TestSuite(
        test_cases=test_cases,
        fixtures=test_suite.fixtures,
        metadata=test_suite.metadata,
    )

    output_filename = f"{status}.yml"
    rasa.utils.io.write_yaml(
        new_test_suite.as_dict(), target=os.path.join(output_dir, output_filename)
    )

    structlogger.info(
        "rasa.e2e_test.save_e2e_test_cases",
        message=f"E2e tests with '{status}' status are written to file "
        f"'{output_filename}'.",
    )
    if status == STATUS_PASSED:
        structlogger.info(
            "rasa.e2e_test.save_e2e_test_cases",
            message=f"You can use the file '{output_filename}' in case you want "
            f"to create training data for fine-tuning an LLM via "
            f"'rasa llm finetune prepare-data'.",
        )


def has_test_case_with_assertions(test_cases: List[TestCase]) -> bool:
    """Check if the test cases contain assertions."""
    try:
        next(test_case for test_case in test_cases if test_case.uses_assertions())
    except StopIteration:
        return False

    return True


def verify_beta_feature_flag_for_assertions(
    test_cases: List[TestCase], beta_flag_verified: bool
) -> bool:
    """Verify the beta feature flag for assertions."""
    if beta_flag_verified:
        return True

    if not has_test_case_with_assertions(test_cases):
        return beta_flag_verified

    try:
        ensure_beta_feature_is_enabled(
            "end-to-end testing with assertions",
            RASA_PRO_BETA_E2E_ASSERTIONS_ENV_VAR_NAME,
        )
    except BetaNotEnabledException as exc:
        rasa.shared.utils.cli.print_error_and_exit(str(exc))

    return True


def _save_tested_commands_histogram(
    count_dict: Dict[str, int], test_status: str, output_dir: str
) -> None:
    """Creates a histogram from a count dictionary and
    saves it to the specified directory.

    Args:
        count_dict (Dict[str, int]): A dictionary where keys are categories
        and values are counts.
        test_status (str): passing or failing
        output_dir (str): The directory path where the histogram
        image will be saved.
    """
    if not count_dict:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(count_dict.keys(), count_dict.values(), color="blue")
    plt.xlabel("Commands")
    plt.ylabel("Counts")
    plt.title("Tested commands histogram")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_filename = f"commands_histogram_for_{test_status}_tests.png"
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path)
    plt.close()

    structlogger.info(
        "rasa.e2e_test._save_tested_commands_histogram",
        message=f"Commands histogram for {test_status} e2e tests"
        f"is written to '{output_filename}'.",
    )
